import platform
import socket
from typing import Optional, Tuple


def get_local_ip() -> str:
    """
    获取适合当前操作系统的本地IP地址。
    在Windows上返回127.0.0.1，在其他系统上返回0.0.0.0。
    
    Returns:
        str: 适合当前操作系统的本地IP地址
    """
    system = platform.system().lower()
    
    if system == "windows":
        # Windows不支持0.0.0.0作为本地调试地址
        return "127.0.0.1"
    else:
        # Linux/Mac等系统支持0.0.0.0
        return "0.0.0.0"


def validate_ip_address(ip: str) -> bool:
    """
    验证IP地址格式是否有效。
    
    Args:
        ip: 要验证的IP地址字符串
        
    Returns:
        bool: IP地址是否有效
    """
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        return False


class LitServer:
    """
    一个简单的服务器类，演示如何根据操作系统选择合适的IP地址。
    """
    
    def __init__(self, host: Optional[str] = None, port: int = 8080):
        """
        初始化服务器。
        
        Args:
            host: 服务器主机地址，如果为None则自动选择
            port: 服务器端口号
            
        Raises:
            ValueError: 如果端口号无效或IP地址格式错误
        """
        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(f"无效的端口号: {port}. 端口号必须在0-65535之间")
        
        self.port = port
        
        # 如果没有指定host，根据操作系统自动选择
        if host is None:
            self.host = get_local_ip()
        else:
            if not validate_ip_address(host):
                raise ValueError(f"无效的IP地址格式: {host}")
            self.host = host
        
        self.server_socket: Optional[socket.socket] = None
    
    def start(self) -> None:
        """
        启动服务器。
        
        Raises:
            RuntimeError: 如果服务器启动失败
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            
            print(f"服务器启动成功:")
            print(f"  主机地址: {self.host}")
            print(f"  端口号: {self.port}")
            print(f"  操作系统: {platform.system()}")
            
            if self.host == "0.0.0.0":
                print("  注意: 在Linux/Mac上，0.0.0.0表示监听所有网络接口")
            else:
                print("  注意: 在Windows上使用127.0.0.1进行本地调试")
                
        except socket.error as e:
            raise RuntimeError(f"服务器启动失败: {e}")
    
    def stop(self) -> None:
        """
        停止服务器。
        """
        if self.server_socket:
            try:
                self.server_socket.close()
                print("服务器已停止")
            except socket.error as e:
                print(f"关闭服务器时出错: {e}")
    
    def get_server_info(self) -> Tuple[str, int]:
        """
        获取服务器信息。
        
        Returns:
            Tuple[str, int]: (主机地址, 端口号)
        """
        return (self.host, self.port)


def main():
    """
    主函数，演示LitServer的使用。
    """
    print("=" * 50)
    print("LitServer 演示程序")
    print("=" * 50)
    
    # 测试1: 自动选择IP地址
    print("\n测试1: 自动选择IP地址")
    server1 = LitServer(port=8080)
    try:
        server1.start()
        print(f"服务器信息: {server1.get_server_info()}")
    except (ValueError, RuntimeError) as e:
        print(f"错误: {e}")
    finally:
        server1.stop()
    
    # 测试2: 手动指定IP地址
    print("\n测试2: 手动指定IP地址")
    server2 = LitServer(host="127.0.0.1", port=8081)
    try:
        server2.start()
        print(f"服务器信息: {server2.get_server_info()}")
    except (ValueError, RuntimeError) as e:
        print(f"错误: {e}")
    finally:
        server2.stop()
    
    # 测试3: 无效的IP地址
    print("\n测试3: 无效的IP地址")
    try:
        server3 = LitServer(host="invalid_ip", port=8082)
        server3.start()
    except ValueError as e:
        print(f"预期的错误: {e}")
    
    # 测试4: 无效的端口号
    print("\n测试4: 无效的端口号")
    try:
        server4 = LitServer(port=70000)
        server4.start()
    except ValueError as e:
        print(f"预期的错误: {e}")
    
    print("\n" + "=" * 50)
    print("演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()


# 测试用例
def test_lit_server():
    """
    测试LitServer类的功能。
    """
    print("\n" + "=" * 50)
    print("运行测试用例")
    print("=" * 50)
    
    # 测试1: Windows系统不应该使用0.0.0.0
    print("\n测试1: Windows系统检测")
    original_system = platform.system
    try:
        # 模拟Windows系统
        platform.system = lambda: "Windows"
        ip = get_local_ip()
        assert ip == "127.0.0.1", f"Windows应该返回127.0.0.1，但返回了{ip}"
        print(f"  ✓ Windows测试通过: 返回 {ip}")
    finally:
        platform.system = original_system
    
    # 测试2: Linux系统可以使用0.0.0.0
    print("\n测试2: Linux系统检测")
    try:
        # 模拟Linux系统
        platform.system = lambda: "Linux"
        ip = get_local_ip()
        assert ip == "0.0.0.0", f"Linux应该返回0.0.0.0，但返回了{ip}"
        print(f"  ✓ Linux测试通过: 返回 {ip}")
    finally:
        platform.system = original_system
    
    # 测试3: IP地址验证
    print("\n测试3: IP地址验证")
    assert validate_ip_address("127.0.0.1") == True
    assert validate_ip_address("0.0.0.0") == True
    assert validate_ip_address("192.168.1.1") == True
    assert validate_ip_address("invalid_ip") == False
    assert validate_ip_address("256.256.256.256") == False
    print("  ✓ IP地址验证测试通过")
    
    # 测试4: 端口号验证
    print("\n测试4: 端口号验证")
    try:
        LitServer(port=-1)
        assert False, "应该抛出ValueError"
    except ValueError:
        print("  ✓ 负数端口号测试通过")
    
    try:
        LitServer(port=70000)
        assert False, "应该抛出ValueError"
    except ValueError:
        print("  ✓ 超大端口号测试通过")
    
    try:
        LitServer(port=0)
        print("  ✓ 端口0测试通过（有效端口）")
    except ValueError:
        assert False, "端口0应该是有效的"
    
    print("\n" + "=" * 50)
    print("所有测试用例通过!")
    print("=" * 50)


if __name__ == "__main__":
    # 运行主程序
    main()
    
    # 运行测试
    test_lit_server()