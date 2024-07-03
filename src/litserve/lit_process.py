import multiprocessing
from multiprocessing import shared_memory, Lock, Manager
from multiprocessing.sharedctypes import Value
import pickle
import time
import signal
import sys


class LitSMQ:
    def __init__(self, name: str, metadata_shm: shared_memory.SharedMemory, data_shm: shared_memory.SharedMemory, lock:Lock):
        self.data_size = data_shm.size
        self.name = name
        self.lock = lock

        self.metadata_shm = metadata_shm
        self.data_shm = data_shm
        self.metadata_buffer = metadata_shm.buf
        self.data_buffer = data_shm.buf

        self.head_index = 0
        self.tail_index = 4

    @staticmethod
    def create(name, data_size=10_000):
        manager = Manager()
        lock = manager.Lock()
        try:
            metadata_shm = shared_memory.SharedMemory(create=True, size=128, name=name + '_metadata')
            # Initialize head and tail to zero
            metadata_shm.buf[0:128] = b'\x00' * 128
        except FileExistsError:
            metadata_shm = shared_memory.SharedMemory(name=name + '_metadata')
        
        try:
            data_shm = shared_memory.SharedMemory(create=True, size=data_size, name=name + '_data')
        except FileExistsError:
            data_shm = shared_memory.SharedMemory(name=name + '_data')

        return lock, LitSMQ(name, metadata_shm=metadata_shm, data_shm=data_shm, lock=lock)

    @staticmethod
    def attach(name, lock):
        try:
            metadata_shm = shared_memory.SharedMemory(name=name + '_metadata')
            data_shm = shared_memory.SharedMemory(name=name + '_data')
            return LitSMQ(name, metadata_shm=metadata_shm, data_shm=data_shm, lock=lock)
        except FileNotFoundError as e:
            print(f"Error attaching shared memory: {e}")
            raise e

    def put(self, item):
        item_bytes = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
        item_size = len(item_bytes)
        if item_size + 4 > self.data_size:
            raise ValueError("Item size exceeds queue capacity")

        with self.lock:
            head = int.from_bytes(self.metadata_buffer[self.head_index:self.head_index+4], 'little')
            tail = int.from_bytes(self.metadata_buffer[self.tail_index:self.tail_index+4], 'little')

            if tail >= head:
                if tail + item_size + 4 > self.data_size:
                    if head <= item_size + 4:
                        raise ValueError("Queue is full")
                    # Wrap around
                    tail = 0
            elif tail + item_size + 4 > head:
                raise ValueError("Queue is full")

            self.data_buffer[tail:tail + 4] = item_size.to_bytes(4, 'little')
            tail += 4
            self.data_buffer[tail:tail + item_size] = item_bytes
            tail += item_size
            self.metadata_buffer[self.tail_index:self.tail_index+4] = tail.to_bytes(4, 'little')

    def get(self):
        with self.lock:
            head = int.from_bytes(self.metadata_buffer[self.head_index:self.head_index+4], 'little')
            tail = int.from_bytes(self.metadata_buffer[self.tail_index:self.tail_index+4], 'little')

            if head == tail:
                return None  # Queue is empty

            item_size = int.from_bytes(self.data_buffer[head:head + 4], 'little')
            head += 4
            item_bytes = self.data_buffer[head:head + item_size]
            head += item_size

            if head == self.data_size:
                head = 0

            self.metadata_buffer[self.head_index:self.head_index+4] = head.to_bytes(4, 'little')

            item = pickle.loads(item_bytes)
            return item

    def close(self):
        self.metadata_shm.close()
        self.data_shm.close()

    def unlink(self):
        self.metadata_shm.unlink()
        self.data_shm.unlink()

    def get_shared_memory_names(self):
        return self.metadata_shm.name, self.data_shm.name



class LitDict:
    def __init__(self, name: str, num_buckets: int = 128, data_size: int = 50_000):
        self.num_buckets = num_buckets
        self.data_size = data_size
        self.name = name

        try:
            self.hashmap_shm = shared_memory.SharedMemory(create=True, size=self.num_buckets * 16, name=self.name + '_hashmap')
            self.data_shm = shared_memory.SharedMemory(create=True, size=self.data_size, name=self.name + '_data')
            # Initialize memory to zero
            self.hashmap_shm.buf[0:self.num_buckets * 16] = b'\x00' * (self.num_buckets * 16)
            self.data_shm.buf[0:self.data_size] = b'\x00' * self.data_size
        except FileExistsError:
            self.hashmap_shm = shared_memory.SharedMemory(name=self.name + '_hashmap')
            self.data_shm = shared_memory.SharedMemory(name=self.name + '_data')

        self.hashmap_buffer = self.hashmap_shm.buf
        self.data_buffer = self.data_shm.buf

        # Initialize a lock for each bucket
        self.bucket_locks = [Lock() for _ in range(self.num_buckets)]

    def _hash_func(self, key):
        return key % self.num_buckets

    def put(self, key, value):
        index = self._hash_func(key)
        with self.bucket_locks[index]:
            bucket_offset = index * 16
            bucket_size = int.from_bytes(self.hashmap_buffer[bucket_offset:bucket_offset+8], 'little')
            bucket_start = int.from_bytes(self.hashmap_buffer[bucket_offset+8:bucket_offset+16], 'little')

            # Deserialize the bucket
            if bucket_size == 0:
                bucket = []
                bucket_start = self._find_empty_space(len(pickle.dumps((key, value), protocol=pickle.HIGHEST_PROTOCOL)) + 8)
            else:
                bucket_data = bytes(self.data_buffer[bucket_start:bucket_start + bucket_size])
                bucket = pickle.loads(bucket_data) if bucket_data.strip(b'\x00') else []

            # Update or add the key-value pair
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    bucket[i] = (key, value)
                    break
            else:
                bucket.append((key, value))

            # Serialize the updated bucket
            bucket_data = pickle.dumps(bucket, protocol=pickle.HIGHEST_PROTOCOL)
            if len(bucket_data) > self.data_size - bucket_start:
                raise ValueError("Data size exceeds shared memory capacity")

            self.data_buffer[bucket_start:bucket_start + len(bucket_data)] = bucket_data
            self.hashmap_buffer[bucket_offset:bucket_offset+8] = len(bucket_data).to_bytes(8, 'little')
            self.hashmap_buffer[bucket_offset+8:bucket_offset+16] = bucket_start.to_bytes(8, 'little')

    def get(self, key):
        index = self._hash_func(key)
        with self.bucket_locks[index]:
            bucket_offset = index * 16
            bucket_size = int.from_bytes(self.hashmap_buffer[bucket_offset:bucket_offset+8], 'little')
            bucket_start = int.from_bytes(self.hashmap_buffer[bucket_offset+8:bucket_offset+16], 'little')

            if bucket_size == 0:
                return None

            bucket_data = bytes(self.data_buffer[bucket_start:bucket_start + bucket_size])
            if not bucket_data.strip(b'\x00'):
                return None
            bucket = pickle.loads(bucket_data)
            for k, v in bucket:
                if k == key:
                    return v
            return None

    def remove(self, key):
        index = self._hash_func(key)
        with self.bucket_locks[index]:
            bucket_offset = index * 16
            bucket_size = int.from_bytes(self.hashmap_buffer[bucket_offset:bucket_offset+8], 'little')
            bucket_start = int.from_bytes(self.hashmap_buffer[bucket_offset+8:bucket_offset+16], 'little')

            if bucket_size == 0:
                return

            bucket_data = bytes(self.data_buffer[bucket_start:bucket_start + bucket_size])
            if not bucket_data.strip(b'\x00'):
                return
            bucket = pickle.loads(bucket_data)
            for i, (k, v) in enumerate(bucket):
                if k == key:
                    del bucket[i]
                    break

            bucket_data = pickle.dumps(bucket, protocol=pickle.HIGHEST_PROTOCOL)
            if len(bucket_data) > self.data_size - bucket_start:
                raise ValueError("Data size exceeds shared memory capacity")

            self.data_buffer[bucket_start:bucket_start + len(bucket_data)] = bucket_data
            self.hashmap_buffer[bucket_offset:bucket_offset+8] = len(bucket_data).to_bytes(8, 'little')
            self.hashmap_buffer[bucket_offset+8:bucket_offset+16] = bucket_start.to_bytes(8, 'little')

    def _find_empty_space(self, size):
        for offset in range(0, self.data_size - size, 8):
            if all(b == 0 for b in self.data_buffer[offset:offset + size]):
                return offset
        raise ValueError("No empty space found in shared memory")

    def close(self):
        self.hashmap_shm.close()
        self.data_shm.close()

    def unlink(self):
        self.hashmap_shm.unlink()
        self.data_shm.unlink()


def cleanup_shared_memory(shm_name):
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.unlink()  # Remove the shared memory from the system
        shm.close()   # Close the shared memory object
        print(f"Unlinked and closed shared memory: {shm_name}")
    except FileNotFoundError:
        print(f"Shared memory {shm_name} not found for cleanup.")

def cleanup_shared_memory_list(shm_names):
    for name in shm_names:
        cleanup_shared_memory(name)

def setup_signal_handlers(shm_names):
    def handle_signal(signal_number, frame):
        print(f"Received signal {signal_number}, cleaning up...")
        cleanup_shared_memory_list(shm_names)
        print("Cleanup complete. Exiting now.")
        SystemExit(0)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl-C
    signal.signal(signal.SIGTERM, handle_signal)  # Handle kill or system shutdown signals
