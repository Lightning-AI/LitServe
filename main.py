import platform
import socket
from typing import Optional


def get_local_ip() -> str:
    """
    Get the local IP address suitable for debugging.
    
    On Windows, returns '127.0.0.1' (localhost) because '0.0.0.0' is invalid
    for local debugging. On other platforms, returns '0.0.0.0' as default.
    
    Returns:
        str: The appropriate IP address for local debugging
    """
    try:
        # Detect the operating system
        system = platform.system().lower()
        
        # On Windows, 0.0.0.0 is not valid for local debugging
        # Use localhost instead
        if system == 'windows':
            return '127.0.0.1'
        
        # On other platforms (Linux, macOS, etc.), 0.0.0.0 is acceptable
        return '0.0.0.0'
        
    except Exception as e:
        # Fallback to localhost if any error occurs during detection
        print(f"Warning: Could not detect platform, falling back to localhost: {e}")
        return '127.0.0.1'


def get_local_ip_with_hostname() -> str:
    """
    Alternative method: Get local IP by resolving hostname.
    This provides the actual network interface IP.
    
    Returns:
        str: The local IP address, or '127.0.0.1' on Windows
    """
    try:
        # Get the hostname
        hostname = socket.gethostname()
        
        # Resolve hostname to IP address
        local_ip = socket.gethostbyname(hostname)
        
        # On Windows, if we get 0.0.0.0, replace with localhost
        if local_ip == '0.0.0.0' and platform.system().lower() == 'windows':
            return '127.0.0.1'
        
        return local_ip
        
    except socket.gaierror:
        # Handle DNS resolution errors
        print("Warning: Could not resolve hostname, using default")
        return get_local_ip()
    except Exception as e:
        # Handle any other errors
        print(f"Warning: Error getting local IP: {e}")
        return get_local_ip()


def format_debug_url(port: int = 8000, use_hostname: bool = False) -> str:
    """
    Format a debug URL with the appropriate IP address.
    
    Args:
        port: The port number for the debug server (default: 8000)
        use_hostname: If True, use actual hostname resolution instead of default
        
    Returns:
        str: Formatted URL like 'http://127.0.0.1:8000' or 'http://0.0.0.0:8000'
    """
    # Validate port number
    if not isinstance(port, int) or port < 1 or port > 65535:
        raise ValueError(f"Invalid port number: {port}. Must be between 1 and 65535.")
    
    # Get the appropriate IP address
    if use_hostname:
        ip_address = get_local_ip_with_hostname()
    else:
        ip_address = get_local_ip()
    
    # Format and return the URL
    return f"http://{ip_address}:{port}"


# Example usage and test cases
def test_ip_address_fix():
    """Test cases for the IP address fix."""
    
    # Test 1: Basic functionality
    print("Test 1: Basic IP detection")
    ip = get_local_ip()
    print(f"  Detected IP: {ip}")
    
    # Test 2: Platform-specific behavior
    print("\nTest 2: Platform detection")
    system = platform.system().lower()
    print(f"  Current platform: {system}")
    
    if system == 'windows':
        assert ip == '127.0.0.1', f"On Windows, expected 127.0.0.1, got {ip}"
        print("  ✓ Windows: IP correctly set to 127.0.0.1")
    else:
        assert ip == '0.0.0.0', f"On non-Windows, expected 0.0.0.0, got {ip}"
        print("  ✓ Non-Windows: IP correctly set to 0.0.0.0")
    
    # Test 3: URL formatting
    print("\nTest 3: URL formatting")
    url = format_debug_url(8080)
    print(f"  Debug URL: {url}")
    assert url.startswith("http://"), "URL should start with http://"
    assert ":8080" in url, "URL should contain the port number"
    print("  ✓ URL formatting works correctly")
    
    # Test 4: Invalid port handling
    print("\nTest 4: Invalid port handling")
    try:
        format_debug_url(70000)  # Invalid port
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    
    # Test 5: Alternative method with hostname
    print("\nTest 5: Hostname resolution method")
    ip_with_hostname = get_local_ip_with_hostname()
    print(f"  IP via hostname: {ip_with_hostname}")
    print("  ✓ Hostname resolution works")
    
    # Test 6: Edge case - port 1 (minimum valid port)
    print("\nTest 6: Edge case - minimum port")
    url_min_port = format_debug_url(1)
    print(f"  URL with port 1: {url_min_port}")
    assert ":1" in url_min_port, "URL should contain port 1"
    print("  ✓ Minimum port handled correctly")
    
    # Test 7: Edge case - port 65535 (maximum valid port)
    print("\nTest 7: Edge case - maximum port")
    url_max_port = format_debug_url(65535)
    print(f"  URL with port 65535: {url_max_port}")
    assert ":65535" in url_max_port, "URL should contain port 65535"
    print("  ✓ Maximum port handled correctly")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    # Run the test cases
    test_ip_address_fix()
    
    # Example: Get the debug URL for a Flask/Django server
    print("\nExample usage:")
    debug_url = format_debug_url(5000)
    print(f"  Debug server URL: {debug_url}")
    print(f"  Open this URL in your browser for local debugging")