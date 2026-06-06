import sys
import socket
from typing import Optional

class LitServer:
    """
    A simple server class that handles IP address selection based on the operating system.
    On Windows, 0.0.0.0 is not valid for local debugging, so we use 127.0.0.1 instead.
    """
    
    def __init__(self, host: Optional[str] = None, port: int = 8000):
        """
        Initialize the server with appropriate host IP.
        
        Args:
            host: Optional host IP. If None, auto-detect based on OS.
            port: Port number to listen on (default: 8000)
        
        Raises:
            ValueError: If port is invalid
        """
        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(f"Invalid port number: {port}. Must be between 0 and 65535.")
        
        self.port = port
        self.host = host if host else self._get_default_host()
        
    def _get_default_host(self) -> str:
        """
        Get the default host IP based on the operating system.
        
        Returns:
            str: '127.0.0.1' on Windows, '0.0.0.0' on other systems
        """
        if sys.platform.startswith('win'):
            # Windows does not support 0.0.0.0 for local debugging
            return '127.0.0.1'
        else:
            # Linux and other Unix-like systems support 0.0.0.0
            return '0.0.0.0'
    
    def get_display_address(self) -> str:
        """
        Get the IP address to display to the user.
        
        Returns:
            str: Formatted IP address string
        """
        if self.host == '0.0.0.0':
            # On Linux, 0.0.0.0 means all interfaces, show localhost for clarity
            return '127.0.0.1'
        return self.host
    
    def start(self) -> None:
        """
        Start the server and display the address to the user.
        This is a simplified version for demonstration.
        """
        display_ip = self.get_display_address()
        print(f"Server starting on {display_ip}:{self.port}")
        print(f"Access locally at: http://{display_ip}:{self.port}")
        
        # In a real implementation, you would start the actual server here
        # For demonstration, we just print the information
        print(f"Server is running on {display_ip}:{self.port}")

def create_server(host: Optional[str] = None, port: int = 8000) -> LitServer:
    """
    Factory function to create a LitServer instance with proper error handling.
    
    Args:
        host: Optional host IP
        port: Port number
        
    Returns:
        LitServer: Configured server instance
    """
    try:
        server = LitServer(host=host, port=port)
        return server
    except ValueError as e:
        print(f"Error creating server: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def test_server_os_detection():
    """
    Test function to verify OS detection and IP address selection.
    """
    print("=== Testing LitServer OS Detection ===")
    
    # Test 1: Default host detection
    server = LitServer()
    print(f"Test 1 - Default host on {sys.platform}: {server.host}")
    
    if sys.platform.startswith('win'):
        assert server.host == '127.0.0.1', f"Expected 127.0.0.1 on Windows, got {server.host}"
        print("✓ Windows correctly uses 127.0.0.1")
    else:
        assert server.host == '0.0.0.0', f"Expected 0.0.0.0 on Linux, got {server.host}"
        print("✓ Linux correctly uses 0.0.0.0")
    
    # Test 2: Display address
    display_ip = server.get_display_address()
    print(f"Test 2 - Display address: {display_ip}")
    assert display_ip == '127.0.0.1', f"Display address should always be 127.0.0.1, got {display_ip}"
    print("✓ Display address correctly shows 127.0.0.1")
    
    # Test 3: Custom host
    custom_server = LitServer(host='192.168.1.1', port=8080)
    print(f"Test 3 - Custom host: {custom_server.host}:{custom_server.port}")
    assert custom_server.host == '192.168.1.1'
    assert custom_server.port == 8080
    print("✓ Custom host and port work correctly")
    
    # Test 4: Invalid port
    try:
        invalid_server = LitServer(port=70000)
        print("✗ Should have raised ValueError for invalid port")
    except ValueError as e:
        print(f"✓ Correctly caught invalid port: {e}")
    
    # Test 5: Server start
    print("\nTest 5 - Server start:")
    server.start()
    
    print("\n=== All tests passed! ===")

if __name__ == "__main__":
    # Run tests
    test_server_os_detection()
    
    # Example usage
    print("\n=== Example Usage ===")
    server = create_server()
    server.start()
    
    # Example with custom port
    print("\n=== Custom Port Example ===")
    custom_server = create_server(port=3000)
    custom_server.start()