"""
跨平台IP地址兼容性解决方案
解决Windows上0.0.0.0无效的问题
"""

import platform
import socket
from typing import Optional, Tuple

class LitServer:
    """
    跨平台兼容的服务器类
    自动检测操作系统并使用合适的IP地址
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        初始化服务器
        
        Args:
            host: 监听地址，默认为0.0.0.0
            port: 监听端口，默认为8080
        """
        self.original_host = host
        self.port = port
        self.display_host = self._get_display_host(host)
        self.listen_host = self._get_listen_host(host)
        
    def _get_display_host(self, host: str) -> str:
        """
        获取用于显示的IP地址
        
        Args:
            host: 原始IP地址
            
        Returns:
            适合显示的IP地址
        """
        if self._is_windows() and host == "0.0.0.0":
            return "127.0.0.1"
        return host
    
    def _get_listen_host(self, host: str) -> str:
        """
        获取用于监听的IP地址
        
        Args:
            host: 原始IP地址
            
        Returns:
            适合监听的IP地址
        """
        if self._is_windows() and host == "0.0.0.0":
            return "127.0.0.1"
        return host
    
    @staticmethod
    def _is_windows() -> bool:
        """
        检测当前操作系统是否为Windows
        
        Returns:
            True如果是Windows系统，否则False
        """
        return platform.system().lower() == "windows"
    
    def start(self) -> None:
        """
        启动服务器
        显示正确的IP地址信息
        """
        print(f"服务器启动在 {self.display_host}:{self.port}")
        print(f"实际监听地址: {self.listen_host}:{self.port}")
        
        try:
            # 创建socket服务器
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.listen_host, self.port))
            server_socket.listen(5)
            
            print(f"服务器正在监听 {self.listen_host}:{self.port}")
            print(f"请访问 http://{self.display_host}:{self.port}")
            
            # 简单的请求处理循环
            while True:
                client_socket, address = server_socket.accept()
                print(f"收到来自 {address} 的连接")
                client_socket.close()
                
        except OSError as e:
            print(f"服务器启动失败: {e}")
            if self._is_windows() and self.listen_host == "0.0.0.0":
                print("提示: 在Windows上请使用127.0.0.1代替0.0.0.0")
        except KeyboardInterrupt:
            print("\n服务器已停止")
        finally:
            server_socket.close()
    
    def get_server_info(self) -> dict:
        """
        获取服务器信息
        
        Returns:
            包含服务器配置信息的字典
        """
        return {
            "original_host": self.original_host,
            "display_host": self.display_host,
            "listen_host": self.listen_host,
            "port": self.port,
            "os": platform.system(),
            "is_windows": self._is_windows()
        }


def create_server(host: str = "0.0.0.0", port: int = 8080) -> LitServer:
    """
    创建跨平台兼容的服务器实例
    
    Args:
        host: 监听地址
        port: 监听端口
        
    Returns:
        LitServer实例
    """
    return LitServer(host, port)


# 测试代码
def test_cross_platform_ip():
    """
    测试跨平台IP地址兼容性
    """
    print("=" * 50)
    print("跨平台IP地址兼容性测试")
    print("=" * 50)
    
    # 测试1: 默认配置
    print("\n测试1: 默认配置 (0.0.0.0:8080)")
    server1 = LitServer()
    info1 = server1.get_server_info()
    print(f"操作系统: {info1['os']}")
    print(f"原始地址: {info1['original_host']}")
    print(f"显示地址: {info1['display_host']}")
    print(f"监听地址: {info1['listen_host']}")
    
    # 验证Windows上不会显示0.0.0.0
    if info1['is_windows']:
        assert info1['display_host'] != "0.0.0.0", "Windows上不应显示0.0.0.0"
        assert info1['listen_host'] == "127.0.0.1", "Windows上应使用127.0.0.1"
        print("✓ Windows兼容性测试通过")
    else:
        assert info1['display_host'] == "0.0.0.0", "Linux上应显示0.0.0.0"
        assert info1['listen_host'] == "0.0.0.0", "Linux上应使用0.0.0.0"
        print("✓ Linux兼容性测试通过")
    
    # 测试2: 自定义IP地址
    print("\n测试2: 自定义IP地址 (192.168.1.1:3000)")
    server2 = LitServer("192.168.1.1", 3000)
    info2 = server2.get_server_info()
    print(f"原始地址: {info2['original_host']}")
    print(f"显示地址: {info2['display_host']}")
    print(f"监听地址: {info2['listen_host']}")
    assert info2['display_host'] == "192.168.1.1", "自定义IP应保持不变"
    assert info2['listen_host'] == "192.168.1.1", "自定义IP应保持不变"
    print("✓ 自定义IP测试通过")
    
    # 测试3: 端口验证
    print("\n测试3: 端口验证")
    server3 = LitServer(port=0)
    info3 = server3.get_server_info()
    print(f"端口: {info3['port']}")
    assert info3['port'] == 0, "端口应保持不变"
    print("✓ 端口测试通过")
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)


if __name__ == "__main__":
    # 运行测试
    test_cross_platform_ip()
    
    print("\n" + "=" * 50)
    print("启动服务器示例")
    print("=" * 50)
    
    # 创建服务器实例
    server = create_server()
    print(f"服务器配置: {server.get_server_info()}")
    
    # 取消注释以下行来实际启动服务器
    # server.start()