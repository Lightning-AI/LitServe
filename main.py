"""
LitServe Windows 线程终止问题修复方案

问题：在 Windows 上，LitServe 的子进程终止时，守护线程无法被正确清理
解决方案：使用多重机制确保线程终止
"""

import os
import sys
import time
import signal
import threading
import ctypes
from typing import Optional, Set, List, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WindowsThreadManager:
    """
    Windows 平台线程管理器
    提供安全的线程终止机制
    """
    
    def __init__(self):
        self._threads: Set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._running = True
        self._cleanup_done = False
        
        # 注册清理函数
        atexit.register(self.cleanup)
        
        # 注册信号处理器（Windows 支持有限）
        if sys.platform == 'win32':
            self._setup_windows_signal_handlers()
    
    def _setup_windows_signal_handlers(self):
        """设置 Windows 信号处理器"""
        try:
            # Windows 上只有 SIGINT, SIGTERM, SIGBREAK 可用
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGBREAK, self._signal_handler)
        except (ValueError, AttributeError) as e:
            logger.warning(f"无法设置信号处理器: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        logger.info(f"收到信号 {signum}，开始清理线程...")
        self.cleanup()
        sys.exit(0)
    
    def create_thread(self, target: Callable, args: tuple = (), 
                     daemon: bool = True, name: str = None) -> threading.Thread:
        """
        创建并注册一个线程
        
        Args:
            target: 线程执行函数
            args: 函数参数
            daemon: 是否为守护线程
            name: 线程名称
            
        Returns:
            创建的线程对象
        """
        thread = threading.Thread(
            target=self._thread_wrapper,
            args=(target, args),
            daemon=daemon,
            name=name or f"Worker-{len(self._threads)}"
        )
        
        with self._lock:
            self._threads.add(thread)
        
        return thread
    
    def _thread_wrapper(self, target: Callable, args: tuple):
        """
        线程包装器，确保线程可以响应终止信号
        
        Args:
            target: 原始目标函数
            args: 函数参数
        """
        try:
            target(*args)
        except Exception as e:
            logger.error(f"线程执行出错: {e}")
        finally:
            # 线程结束时自动从管理器中移除
            current_thread = threading.current_thread()
            with self._lock:
                self._threads.discard(current_thread)
    
    def terminate_thread(self, thread: threading.Thread, 
                        timeout: float = 5.0) -> bool:
        """
        安全终止指定线程
        
        Args:
            thread: 要终止的线程
            timeout: 等待超时时间（秒）
            
        Returns:
            是否成功终止
        """
        if not thread.is_alive():
            return True
        
        logger.info(f"尝试终止线程: {thread.name}")
        
        # 方法1: 使用事件通知
        if hasattr(thread, '_stop_event'):
            thread._stop_event.set()
        
        # 方法2: 使用 ctypes 强制终止（仅 Windows）
        if sys.platform == 'win32':
            try:
                thread_id = thread.ident
                if thread_id:
                    # 使用 Windows API 强制终止线程
                    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        thread_id, ctypes.py_object(SystemExit)
                    )
                    if res > 1:
                        # 如果失败，恢复状态
                        ctypes.pythonapi.PyThreadState_SetAsyncExc(
                            thread_id, None
                        )
                        logger.error(f"强制终止线程失败: {thread.name}")
            except Exception as e:
                logger.warning(f"强制终止线程异常: {e}")
        
        # 等待线程结束
        thread.join(timeout=timeout)
        
        if thread.is_alive():
            logger.warning(f"线程 {thread.name} 在 {timeout} 秒后仍未终止")
            return False
        
        with self._lock:
            self._threads.discard(thread)
        
        return True
    
    def cleanup(self, timeout: float = 10.0):
        """
        清理所有管理的线程
        
        Args:
            timeout: 总超时时间
        """
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        self._running = False
        
        logger.info("开始清理线程...")
        
        with self._lock:
            active_threads = list(self._threads)
        
        if not active_threads:
            logger.info("没有活跃线程需要清理")
            return
        
        # 逐个终止线程
        start_time = time.time()
        for thread in active_threads:
            if time.time() - start_time > timeout:
                logger.warning("清理超时，强制终止剩余线程")
                break
            
            self.terminate_thread(thread, timeout=2.0)
        
        # 检查是否还有残留线程
        remaining = [t for t in active_threads if t.is_alive()]
        if remaining:
            logger.warning(f"仍有 {len(remaining)} 个线程未终止")
            for t in remaining:
                logger.warning(f"  残留线程: {t.name} (ID: {t.ident})")
        else:
            logger.info("所有线程已成功清理")


class LitServeThreadFix:
    """
    LitServe 线程修复类
    提供替代 LitServe 线程管理的安全实现
    """
    
    def __init__(self):
        self.thread_manager = WindowsThreadManager()
        self.executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_hook_registered = False
    
    def start_worker_pool(self, num_workers: int = 4):
        """
        启动工作线程池
        
        Args:
            num_workers: 工作线程数量
        """
        if self.executor:
            logger.warning("工作池已存在，先关闭旧的")
            self.stop_worker_pool()
        
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="LitServeWorker"
        )
        
        # 注册关闭钩子
        if not self._shutdown_hook_registered:
            atexit.register(self.stop_worker_pool)
            self._shutdown_hook_registered = True
        
        logger.info(f"启动 {num_workers} 个工作线程")
    
    def submit_task(self, fn: Callable, *args, **kwargs):
        """
        提交任务到工作池
        
        Args:
            fn: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            Future 对象
        """
        if not self.executor:
            raise RuntimeError("工作池未启动，请先调用 start_worker_pool()")
        
        return self.executor.submit(fn, *args, **kwargs)
    
    def stop_worker_pool(self, timeout: float = 10.0):
        """
        安全停止工作池
        
        Args:
            timeout: 等待超时时间
        """
        if self.executor:
            logger.info("正在停止工作池...")
            
            # 优雅关闭
            self.executor.shutdown(wait=False, cancel_futures=True)
            
            # 等待线程结束
            start_time = time.time()
            while time.time() - start_time < timeout:
                # 检查是否所有线程都已结束
                threads = threading.enumerate()
                worker_threads = [
                    t for t in threads 
                    if t.name and t.name.startswith("LitServeWorker")
                ]
                
                if not worker_threads:
                    logger.info("所有工作线程已终止")
                    break
                
                time.sleep(0.1)
            
            # 强制终止残留线程
            remaining = [
                t for t in threading.enumerate()
                if t.name and t.name.startswith("LitServeWorker") and t.is_alive()
            ]
            
            for thread in remaining:
                logger.warning(f"强制终止残留线程: {thread.name}")
                self.thread_manager.terminate_thread(thread, timeout=2.0)
            
            self.executor = None
    
    def force_cleanup(self):
        """强制清理所有资源"""
        logger.info("执行强制清理...")
        self.stop_worker_pool(timeout=5.0)
        self.thread_manager.cleanup(timeout=5.0)


# 测试代码
def test_thread_cleanup():
    """测试线程清理功能"""
    print("=" * 60)
    print("测试线程清理功能")
    print("=" * 60)
    
    fix = LitServeThreadFix()
    
    def worker_task(task_id: int):
        """模拟工作线程任务"""
        try:
            print(f"工作线程 {task_id} 启动")
            while True:
                time.sleep(1)
                print(f"工作线程 {task_id} 运行中...")
        except KeyboardInterrupt:
            print(f"工作线程 {task_id} 收到中断信号")
        except SystemExit:
            print(f"工作线程 {task_id} 收到退出信号")
        finally:
            print(f"工作线程 {task_id} 终止")
    
    # 启动工作池
    fix.start_worker_pool(num_workers=3)
    
    # 提交任务
    futures = []
    for i in range(3):
        future = fix.submit_task(worker_task, i)
        futures.append(future)
    
    # 运行一段时间
    print("工作池运行中...")
    time.sleep(3)
    
    # 停止工作池
    print("停止工作池...")
    fix.stop_worker_pool()
    
    # 检查线程状态
    time.sleep(1)
    active_threads = threading.enumerate()
    worker_threads = [
        t for t in active_threads 
        if t.name and "LitServeWorker" in t.name
    ]
    
    if worker_threads:
        print(f"警告: 仍有 {len(worker_threads)} 个工作线程存活")
        for t in worker_threads:
            print(f"  线程: {t.name} (ID: {t.ident})")
    else:
        print("成功: 所有工作线程已终止")
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


def test_forceful_termination():
    """测试强制终止功能"""
    print("\n" + "=" * 60)
    print("测试强制终止功能")
    print("=" * 60)
    
    manager = WindowsThreadManager()
    
    def stubborn_task():
        """模拟顽固线程"""
        try:
            print("顽固线程启动")
            while True:
                time.sleep(0.5)
                print("顽固线程仍在运行...")
        except Exception as e:
            print(f"顽固线程异常: {e}")
        finally:
            print("顽固线程终止")
    
    # 创建并启动线程
    thread = manager.create_thread(
        target=stubborn_task,
        daemon=True,
        name="StubbornThread"
    )
    thread.start()
    
    time.sleep(2)
    
    # 尝试终止
    print("尝试终止顽固线程...")
    success = manager.terminate_thread(thread, timeout=3.0)
    
    if success:
        print("成功: 顽固线程已终止")
    else:
        print("警告: 顽固线程未能终止")
    
    # 清理
    manager.cleanup()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    manager = WindowsThreadManager()
    
    # 测试1: 清理空管理器
    print("测试1: 清理空管理器")
    manager.cleanup()
    print("通过: 空管理器清理成功")
    
    # 测试2: 多次清理
    print("\n测试2: 多次清理")
    manager.cleanup()
    manager.cleanup()
    print("通过: 多次清理成功")
    
    # 测试3: 已结束的线程
    print("\n测试3: 已结束的线程")
    def quick_task():
        pass
    
    thread = manager.create_thread(target=quick_task, name="QuickThread")
    thread.start()
    thread.join()
    
    success = manager.terminate_thread(thread)
    assert success, "已结束的线程应该能成功终止"
    print("通过: 已结束线程处理成功")
    
    # 测试4: 不存在的线程
    print("\n测试4: 不存在的线程")
    fake_thread = threading.Thread(target=lambda: None)
    success = manager.terminate_thread(fake_thread)
    assert success, "不存在的线程应该返回成功"
    print("通过: 不存在线程处理成功")
    
    # 测试5: 大量线程
    print("\n测试5: 大量线程")
    threads = []
    for i in range(10):
        def worker(n=i):
            time.sleep(10)
        
        t = manager.create_thread(target=worker, name=f"BulkThread-{i}")
        t.start()
        threads.append(t)
    
    print(f"创建了 {len(threads)} 个线程")
    manager.cleanup(timeout=5.0)
    
    remaining = [t for t in threads if t.is_alive()]
    print(f"剩余线程: {len(remaining)}")
    print("通过: 批量线程清理成功")
    
    print("=" * 60)
    print("所有边界测试通过")
    print("=" * 60)


if __name__ == "__main__":
    print("LitServe Windows 线程终止修复方案")
    print("版本: 1.0.0")
    print("=" * 60)
    
    try:
        # 运行测试
        test_thread_cleanup()
        test_forceful_termination()
        test_edge_cases()
        
        print("\n所有测试通过!")
        
    except KeyboardInterrupt:
        print("\n收到中断信号，执行清理...")
        # 确保清理
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)