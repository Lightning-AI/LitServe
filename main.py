"""
跨平台IP地址处理模块
解决Windows上0.0.0.0无效的问题，自动检测操作系统并选择合适的IP地址
"""

import socket
import platform
from typing import Optional, Tuple


def get_local_ip() -> str:
    """
    获取本机实际IP地址
    
    Returns:
        str: 本机IP地址，如果获取失败则返回127.0.0.1
    """
    try:
        # 创建一个UDP套接字来获取本机IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # 连接到一个外部地址（实际上不会发送数据）
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        # 如果获取失败，返回回环地址
        return "127.0.0.1"


def get_bind_address(prefer_public: bool = False) -> str:
    """
    获取适合绑定的IP地址
    
    Args:
        prefer_public: 是否优先使用公共IP（0.0.0.0），仅在Linux上有效
    
    Returns:
        str: 适合当前操作系统的绑定地址
        
    Raises:
        ValueError: 如果参数无效
    """
    if not isinstance(prefer_public, bool):
        raise ValueError("prefer_public 必须是布尔值")
    
    system = platform.system().lower()
    
    if system == "windows":
        # Windows不支持0.0.0.0，使用127.0.0.1或实际IP
        return "127.0.0.1"
    elif system in ("linux", "darwin"):  # darwin 是 macOS
        if prefer_public:
            return "0.0.0.0"
        else:
            return "127.0.0.1"
    else:
        # 其他系统，保守使用127.0.0.1
        return "127.0.0.1"


def get_display_address(bind_address: str) -> str:
    """
    获取用于显示给用户的IP地址
    
    Args:
        bind_address: 绑定的IP地址
    
    Returns:
        str: 用户可访问的IP地址
    """
    if bind_address == "0.0.0.0":
        # 如果绑定到所有接口，显示本机实际IP
        return get_local_ip()
    else:
        return bind_address


class LitServer:
    """
    简单的服务器类，演示跨平台IP处理
    """
    
    def __init__(self, host: Optional[str] = None, port: int = 8080):
        """
        初始化服务器
        
        Args:
            host: 绑定的主机地址，如果为None则自动选择
            port: 端口号
            
        Raises:
            ValueError: 如果端口号无效
        """
        if not isinstance(port, int) or port < 1 or port > 65535:
            raise ValueError("端口号必须在1-65535之间")
        
        self.port = port
        
        if host is None:
            # 自动选择地址
            self.bind_address = get_bind_address(prefer_public=False)
        else:
            # 验证用户提供的地址
            if not self._validate_ip(host):
                raise ValueError(f"无效的IP地址: {host}")
            self.bind_address = host
        
        # 获取用于显示的地址
        self.display_address = get_display_address(self.bind_address)
    
    @staticmethod
    def _validate_ip(ip: str) -> bool:
        """
        验证IP地址格式
        
        Args:
            ip: 要验证的IP地址字符串
        
        Returns:
            bool: 是否为有效的IP地址
        """
        try:
            socket.inet_aton(ip)
            return True
        except socket.error:
            return False
    
    def get_url(self) -> str:
        """
        获取服务器访问URL
        
        Returns:
            str: 完整的访问URL
        """
        return f"http://{self.display_address}:{self.port}"
    
    def start(self) -> None:
        """
        启动服务器（模拟）
        """
        print(f"服务器启动在: {self.get_url()}")
        print(f"绑定地址: {self.bind_address}")
        print(f"操作系统: {platform.system()}")


def test_cross_platform_ip():
    """
    测试跨平台IP处理功能
    """
    print("=" * 50)
    print("跨平台IP地址测试")
    print("=" * 50)
    
    # 测试1: 自动选择地址
    print("\n1. 测试自动选择地址:")
    server = LitServer()
    print(f"   URL: {server.get_url()}")
    print(f"   绑定地址: {server.bind_address}")
    
    # 测试2: 指定地址
    print("\n2. 测试指定地址:")
    try:
        server2 = LitServer(host="127.0.0.1", port=3000)
        print(f"   URL: {server2.get_url()}")
    except ValueError as e:
        print(f"   错误: {e}")
    
    # 测试3: 无效地址
    print("\n3. 测试无效地址:")
    try:
        server3 = LitServer(host="invalid.ip")
        print(f"   URL: {server3.get_url()}")
    except ValueError as e:
        print(f"   错误: {e}")
    
    # 测试4: 无效端口
    print("\n4. 测试无效端口:")
    try:
        server4 = LitServer(port=99999)
        print(f"   URL: {server4.get_url()}")
    except ValueError as e:
        print(f"   错误: {e}")
    
    # 测试5: 显示地址功能
    print("\n5. 测试显示地址:")
    display_addr = get_display_address("0.0.0.0")
    print(f"   绑定0.0.0.0时显示: {display_addr}")
    display_addr2 = get_display_address("127.0.0.1")
    print(f"   绑定127.0.0.1时显示: {display_addr2}")
    
    # 测试6: 操作系统检测
    print("\n6. 测试操作系统检测:")
    system = platform.system()
    print(f"   当前系统: {system}")
    bind_addr = get_bind_address(prefer_public=True)
    print(f"   推荐绑定地址: {bind_addr}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


def main():
    """
    主函数 - 演示服务器启动
    """
    print("LitServer 演示")
    print("-" * 30)
    
    # 创建并启动服务器
    server = LitServer()
    server.start()
    
    # 运行测试
    print("\n运行测试...")
    test_cross_platform_ip()


if __name__ == "__main__":
    main()