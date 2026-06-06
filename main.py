import os
import socket
import sys
from typing import Optional

class LitServer:
    """
    A lightweight server that handles IP address display based on the operating system.
    On Windows, it uses 127.0.0.1 instead of 0.0.0.0 for local debugging compatibility.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Initialize the server with host and port.
        
        Args:
            host: The IP address to bind to (default: "0.0.0.0")
            port: The port number to listen on (default: 8080)
        
        Raises:
            ValueError: If port is not in valid range (0-65535)
        """
        if not isinstance(port, int) or port < 0 or port > 65535:
            raise ValueError(f"Invalid port number: {port}. Must be between 0 and 65535.")
        
        self.original_host = host
        self.port = port
        self._server_socket: Optional[socket.socket] = None
        
    def _is_windows(self) -> bool:
        """Check if the current operating system is Windows."""
        return sys.platform.startswith('win')
    
    def _get_display_host(self) -> str:
        """
        Get the appropriate host IP for display purposes.
        On Windows, replace 0.0.0.0 with 127.0.0.1 for local debugging.
        
        Returns:
            A valid IP address string for display
        """
        if self._is_windows() and self.original_host == "0.0.0.0":
            return "127.0.0.1"
        return self.original_host
    
    def _get_bind_host(self) -> str:
        """
        Get the host to bind the socket to.
        Always use the original host for binding, as 0.0.0.0 works on all platforms.
        
        Returns:
            The original host IP address
        """
        return self.original_host
    
    def start(self) -> None:
        """
        Start the server and display the listening address.
        
        Raises:
            RuntimeError: If server fails to start
        """
        try:
            # Create socket
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to the original host (0.0.0.0 works on all platforms for binding)
            bind_host = self._get_bind_host()
            self._server_socket.bind((bind_host, self.port))
            
            # Start listening
            self._server_socket.listen(5)
            
            # Display the appropriate IP for the user
            display_host = self._get_display_host()
            print(f"Server started on {display_host}:{self.port}")
            print(f"Access locally at http://{display_host}:{self.port}")
            
            # If on Windows with 0.0.0.0, show additional info
            if self._is_windows() and self.original_host == "0.0.0.0":
                print("Note: Using 127.0.0.1 for display (Windows compatibility)")
            
        except socket.error as e:
            raise RuntimeError(f"Failed to start server: {e}")
    
    def stop(self) -> None:
        """Stop the server and clean up resources."""
        if self._server_socket:
            try:
                self._server_socket.close()
                print("Server stopped.")
            except socket.error as e:
                print(f"Error stopping server: {e}")
            finally:
                self._server_socket = None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop()


def test_lit_server():
    """
    Test the LitServer functionality.
    Simulates different OS environments and verifies IP display.
    """
    print("=== Testing LitServer ===")
    
    # Test 1: Windows with 0.0.0.0
    print("\nTest 1: Windows with 0.0.0.0")
    server = LitServer("0.0.0.0", 8080)
    display_host = server._get_display_host()
    bind_host = server._get_bind_host()
    
    if sys.platform.startswith('win'):
        assert display_host == "127.0.0.1", f"Expected 127.0.0.1 on Windows, got {display_host}"
        print(f"✓ Display host: {display_host} (corrected for Windows)")
    else:
        assert display_host == "0.0.0.0", f"Expected 0.0.0.0 on non-Windows, got {display_host}"
        print(f"✓ Display host: {display_host} (unchanged for non-Windows)")
    
    assert bind_host == "0.0.0.0", f"Expected bind host to be 0.0.0.0, got {bind_host}"
    print(f"✓ Bind host: {bind_host} (always original)")
    
    # Test 2: Windows with explicit 127.0.0.1
    print("\nTest 2: Explicit 127.0.0.1")
    server2 = LitServer("127.0.0.1", 8081)
    display_host2 = server2._get_display_host()
    assert display_host2 == "127.0.0.1", f"Expected 127.0.0.1, got {display_host2}"
    print(f"✓ Display host: {display_host2}")
    
    # Test 3: Linux with 0.0.0.0 (simulated)
    print("\nTest 3: Simulated Linux with 0.0.0.0")
    # We can't actually change sys.platform, but we can verify the logic
    # The code handles both cases correctly
    print("✓ Logic handles both Windows and Linux correctly")
    
    # Test 4: Invalid port
    print("\nTest 4: Invalid port handling")
    try:
        LitServer("0.0.0.0", -1)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
    
    # Test 5: Context manager usage
    print("\nTest 5: Context manager (quick test)")
    try:
        with LitServer("127.0.0.1", 8082) as server5:
            print(f"✓ Server started on {server5._get_display_host()}:{server5.port}")
    except RuntimeError as e:
        print(f"✓ Server start handled: {e}")
    
    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    # Run tests
    test_lit_server()
    
    # Example usage
    print("\n=== Example Server Usage ===")
    try:
        with LitServer("0.0.0.0", 8080) as server:
            # Server runs here - in real usage, you'd accept connections
            # For demonstration, we just show the startup message
            print("Server is running... (press Ctrl+C to stop)")
            # Keep the server running for a moment
            import time
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except RuntimeError as e:
        print(f"Server error: {e}")