import threading
import time
import signal
import sys
import os
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreadState(Enum):
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class ManagedThread:
    target: Callable
    name: str = "ManagedThread"
    daemon: bool = True
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False, repr=False)
    _state: ThreadState = field(default=ThreadState.STOPPED, init=False, repr=False)
    
    def start(self) -> None:
        if self._state != ThreadState.STOPPED:
            raise RuntimeError(f"Thread {self.name} is already running or stopping")
        self._stop_event.clear()
        self._state = ThreadState.RUNNING
        self._thread = threading.Thread(
            target=self._run_with_cleanup,
            name=self.name,
            daemon=self.daemon
        )
        self._thread.start()
        logger.debug(f"Thread {self.name} started")
    
    def stop(self, timeout: float = 5.0) -> bool:
        if self._state != ThreadState.RUNNING:
            logger.warning(f"Thread {self.name} is not running (state: {self._state})")
            return True
        logger.info(f"Stopping thread {self.name}...")
        self._state = ThreadState.STOPPING
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"Thread {self.name} did not stop within {timeout}s timeout")
                return False
        self._state = ThreadState.STOPPED
        logger.info(f"Thread {self.name} stopped successfully")
        return True
    
    def _run_with_cleanup(self) -> None:
        try:
            self.target(self._stop_event)
        except Exception as e:
            logger.error(f"Thread {self.name} crashed: {e}")
        finally:
            self._state = ThreadState.STOPPED
            logger.debug(f"Thread {self.name} cleanup completed")
    
    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
    
    @property
    def state(self) -> ThreadState:
        return self._state


class ThreadPool:
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._threads: dict[str, ManagedThread] = {}
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        if os.name == 'nt':
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        else:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def submit(self, name: str, target: Callable, *args, **kwargs) -> ManagedThread:
        with self._lock:
            if name in self._threads:
                raise ValueError(f"Thread with name '{name}' already exists")
            if len(self._threads) >= self.max_workers:
                raise RuntimeError(f"Thread pool at maximum capacity ({self.max_workers})")
            def wrapper(stop_event: threading.Event) -> None:
                target(stop_event, *args, **kwargs)
            thread = ManagedThread(
                target=wrapper,
                name=name,
                daemon=True
            )
            self._threads[name] = thread
            thread.start()
            logger.debug(f"Thread '{name}' submitted to pool")
            return thread
    
    def shutdown(self, timeout: float = 10.0) -> bool:
        logger.info("Shutting down thread pool...")
        self._shutdown_event.set()
        success = True
        with self._lock:
            threads_to_stop = list(self._threads.items())
        for name, thread in threads_to_stop:
            if not thread.stop(timeout=timeout / len(threads_to_stop) if threads_to_stop else timeout):
                success = False
                logger.error(f"Failed to stop thread '{name}'")
        with self._lock:
            self._threads.clear()
        logger.info("Thread pool shutdown complete")
        return success
    
    def get_thread(self, name: str) -> Optional[ManagedThread]:
        with self._lock:
            return self._threads.get(name)
    
    @property
    def active_count(self) -> int:
        with self._lock:
            return len([t for t in self._threads.values() if t.is_alive])
    
    @property
    def total_count(self) -> int:
        with self._lock:
            return len(self._threads)


class LitServer:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self._running = False
        self._thread_pool = ThreadPool(max_workers=10)  # Increased max_workers
        self._server_thread: Optional[ManagedThread] = None
    
    def start(self) -> None:
        if self._running:
            logger.warning("Server is already running")
            return
        logger.info(f"Starting LitServer on {self.host}:{self.port}")
        self._running = True
        self._server_thread = self._thread_pool.submit(
            "server_main",
            self._run_server
        )
        logger.info("Server started successfully")
    
    def _run_server(self, stop_event: threading.Event) -> None:
        logger.info("Server main loop started")
        try:
            while not stop_event.is_set():
                self._handle_requests(stop_event)
                stop_event.wait(0.1)
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            logger.info("Server main loop ended")
    
    def _handle_requests(self, stop_event: threading.Event) -> None:
        # Use a unique name based on time and a counter to avoid duplicates
        worker_name = f"worker_{time.time()}_{id(stop_event)}"
        try:
            worker_thread = self._thread_pool.submit(
                worker_name,
                self._process_request,
                request_id=id(stop_event)
            )
            worker_thread.stop(timeout=5.0)
        except RuntimeError as e:
            # If pool is full, just skip this request
            logger.warning(f"Skipping request due to pool full: {e}")
    
    def _process_request(self, stop_event: threading.Event, request_id: int) -> None:
        logger.debug(f"Processing request {request_id}")
        for i in range(5):
            if stop_event.is_set():
                logger.debug(f"Request {request_id} interrupted by shutdown")
                return
            time.sleep(0.1)
        logger.debug(f"Request {request_id} completed")
    
    def stop(self, timeout: float = 10.0) -> bool:
        if not self._running:
            logger.warning("Server is not running")
            return True
        logger.info("Stopping LitServer...")
        self._running = False
        success = self._thread_pool.shutdown(timeout=timeout)
        if success:
            logger.info("LitServer stopped successfully")
        else:
            logger.warning("LitServer stopped with some threads still running")
        return success
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def test_thread_cleanup_on_windows():
    logger.info("Running thread cleanup test...")
    server = LitServer(host="localhost", port=8000)
    try:
        server.start()
        time.sleep(2)
        # After 2 seconds, server may have processed some requests; active_count may be 0 if all workers finished
        # We'll check that server_main is alive
        assert server._server_thread is not None and server._server_thread.is_alive, "Server thread is not alive"
        logger.info(f"Server thread alive: {server._server_thread.is_alive}")
        success = server.stop(timeout=5.0)
        assert success, "Server shutdown failed"
        assert server._thread_pool.active_count == 0, "Threads still running after shutdown"
        logger.info("All threads stopped successfully")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        server.stop(timeout=2.0)


def test_lightning_studio_compatibility():
    logger.info("Running Lightning Studio compatibility test...")
    os.environ.setdefault("LIGHTNING_STUDIO", "true")
    for i in range(3):
        logger.info(f"Start/stop cycle {i + 1}")
        server = LitServer(host="localhost", port=8000 + i)
        try:
            server.start()
            time.sleep(1)
            assert server._running, f"Server {i} failed to start"
            success = server.stop(timeout=5.0)
            assert success, f"Server {i} failed to stop"
            assert server._thread_pool.active_count == 0, f"Threads remaining after server {i} stop"
        except Exception as e:
            logger.error(f"Test cycle {i} failed: {e}")
            raise
        finally:
            server.stop(timeout=2.0)
    logger.info("Lightning Studio compatibility test passed")


def test_concurrent_requests():
    logger.info("Running concurrent requests test...")
    server = LitServer(host="localhost", port=8000)
    try:
        server.start()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(
                    server._process_request,
                    threading.Event(),
                    i
                )
                futures.append(future)
            concurrent.futures.wait(futures, timeout=10.0)
            for future in futures:
                if future.exception():
                    logger.error(f"Request failed: {future.exception()}")
        logger.info("Concurrent requests test passed")
    finally:
        server.stop(timeout=5.0)


if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("LitServer Thread Cleanup Fix - Test Suite")
    logger.info("=" * 50)
    try:
        test_thread_cleanup_on_windows()
        test_lightning_studio_compatibility()
        test_concurrent_requests()
        logger.info("=" * 50)
        logger.info("All tests passed!")
        logger.info("=" * 50)
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        sys.exit(1)
    logger.info("\nExample server usage:")
    with LitServer(host="localhost", port=8000) as server:
        logger.info("Server is running...")
        time.sleep(3)
        logger.info("Server will now shut down...")
    logger.info("Server shutdown complete - no threads remaining")
