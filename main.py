import sys
import platform
from typing import Optional


def get_default_host() -> str:
    """
    根据操作系统返回合适的默认主机地址。
    
    Windows上使用127.0.0.1（因为0.0.0.0在Windows上对用户不友好），
    Linux上使用0.0.0.0（监听所有网络接口）。
    
    Returns:
        str: 默认主机地址
    """
    system = platform.system().lower()
    
    if system == "windows":
        # Windows上使用localhost或127.0.0.1
        # 避免显示0.0.0.0给用户
        return "127.0.0.1"
    else:
        # Linux/Mac上使用0.0.0.0监听所有接口
        return "0.0.0.0"


def get_display_host(host: Optional[str] = None) -> str:
    """
    获取用于显示给用户的友好主机地址。
    
    在Windows上，如果host是0.0.0.0，则替换为127.0.0.1。
    在其他系统上保持原样。
    
    Args:
        host: 原始主机地址，如果为None则使用默认值
        
    Returns:
        str: 用于显示的主机地址
    """
    if host is None:
        host = get_default_host()
    
    system = platform.system().lower()
    
    if system == "windows" and host == "0.0.0.0":
        # Windows上0.0.0.0对用户无效，替换为127.0.0.1
        return "127.0.0.1"
    
    return host


class LitServer:
    """
    模拟的LitServe服务器类，演示主机地址处理。
    """
    
    def __init__(self, host: Optional[str] = None, port: int = 8000):
        """
        初始化服务器。
        
        Args:
            host: 主机地址，如果为None则根据操作系统自动选择
            port: 端口号
            
        Raises:
            ValueError: 如果端口号无效
        """
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError(f"无效的端口号: {port}，端口号必须在1-65535之间")
        
        self.host = host if host is not None else get_default_host()
        self.port = port
        self._running = False
    
    def start(self) -> None:
        """
        启动服务器并显示连接信息。
        
        在Windows上，如果host是0.0.0.0，会显示127.0.0.1给用户。
        """
        display_host = get_display_host(self.host)
        
        print(f"服务器启动中...")
        print(f"监听地址: {self.host}:{self.port}")
        print(f"访问地址: http://{display_host}:{self.port}")
        
        # 实际服务器启动逻辑（这里只是模拟）
        self._running = True
        
        if platform.system().lower() == "windows" and self.host == "0.0.0.0":
            print("注意: 在Windows上，0.0.0.0已被替换为127.0.0.1用于显示")
    
    def stop(self) -> None:
        """停止服务器。"""
        self._running = False
        print("服务器已停止")
    
    def is_running(self) -> bool:
        """检查服务器是否在运行。"""
        return self._running


def main():
    """
    主函数，演示不同操作系统下的行为。
    """
    print("=" * 50)
    print("LitServe 主机地址处理演示")
    print("=" * 50)
    
    # 测试1: 使用默认主机地址
    print("\n1. 使用默认主机地址:")
    server1 = LitServer()
    server1.start()
    server1.stop()
    
    # 测试2: 明确指定0.0.0.0
    print("\n2. 明确指定0.0.0.0:")
    server2 = LitServer(host="0.0.0.0")
    server2.start()
    server2.stop()
    
    # 测试3: 指定其他地址
    print("\n3. 指定其他地址:")
    server3 = LitServer(host="192.168.1.100")
    server3.start()
    server3.stop()
    
    # 测试4: 错误处理 - 无效端口
    print("\n4. 错误处理测试:")
    try:
        server4 = LitServer(port=99999)
    except ValueError as e:
        print(f"捕获到错误: {e}")
    
    print("\n" + "=" * 50)
    print(f"当前操作系统: {platform.system()} {platform.release()}")
    print(f"默认主机地址: {get_default_host()}")
    print("=" * 50)


if __name__ == "__main__":
    main()


# ============ 测试用例 ============
def test_get_default_host():
    """测试默认主机地址获取函数。"""
    import unittest
    
    class TestHostFunctions(unittest.TestCase):
        def test_windows_default_host(self):
            """Windows上默认应该是127.0.0.1"""
            # 模拟Windows环境
            original_system = platform.system
            platform.system = lambda: "Windows"
            
            try:
                host = get_default_host()
                self.assertEqual(host, "127.0.0.1")
            finally:
                platform.system = original_system
        
        def test_linux_default_host(self):
            """Linux上默认应该是0.0.0.0"""
            # 模拟Linux环境
            original_system = platform.system
            platform.system = lambda: "Linux"
            
            try:
                host = get_default_host()
                self.assertEqual(host, "0.0.0.0")
            finally:
                platform.system = original_system
        
        def test_display_host_windows_with_0000(self):
            """Windows上0.0.0.0应该显示为127.0.0.1"""
            original_system = platform.system
            platform.system = lambda: "Windows"
            
            try:
                display = get_display_host("0.0.0.0")
                self.assertEqual(display, "127.0.0.1")
            finally:
                platform.system = original_system
        
        def test_display_host_linux_with_0000(self):
            """Linux上0.0.0.0应该保持不变"""
            original_system = platform.system
            platform.system = lambda: "Linux"
            
            try:
                display = get_display_host("0.0.0.0")
                self.assertEqual(display, "0.0.0.0")
            finally:
                platform.system = original_system
        
        def test_display_host_with_specific_ip(self):
            """指定IP地址应该保持不变"""
            original_system = platform.system
            platform.system = lambda: "Windows"
            
            try:
                display = get_display_host("192.168.1.1")
                self.assertEqual(display, "192.168.1.1")
            finally:
                platform.system = original_system
        
        def test_invalid_port(self):
            """无效端口应该抛出异常"""
            with self.assertRaises(ValueError):
                LitServer(port=0)
            with self.assertRaises(ValueError):
                LitServer(port=65536)
            with self.assertRaises(ValueError):
                LitServer(port=-1)
    
    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestHostFunctions)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 取消注释以下行来运行测试
    # test_get_default_host()