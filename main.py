import platform
import socket
from typing import Optional


def get_display_ip(ip_address: str) -> str:
    """
    Replace 0.0.0.0 with localhost/127.0.0.1 on Windows for better user experience.
    
    Args:
        ip_address: The IP address to check and potentially replace
        
    Returns:
        str: The display-friendly IP address
        
    Raises:
        ValueError: If ip_address is not a valid string
    """
    if not isinstance(ip_address, str):
        raise ValueError("IP address must be a string")
    
    if not ip_address:
        raise ValueError("IP address cannot be empty")
    
    # Only replace 0.0.0.0 on Windows systems
    if ip_address == "0.0.0.0" and platform.system().lower() == "windows":
        return "127.0.0.1"
    
    return ip_address


def get_local_ip() -> Optional[str]:
    """
    Get the local machine's IP address.
    
    Returns:
        Optional[str]: The local IP address, or None if unable to determine
    """
    try:
        # Create a socket connection to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to a public DNS server (doesn't actually send data)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            return local_ip
    except Exception as e:
        print(f"Warning: Unable to determine local IP: {e}")
        return None


def format_server_url(host: str, port: int, use_localhost: bool = True) -> str:
    """
    Format a server URL with proper display IP.
    
    Args:
        host: The host IP address
        port: The port number
        use_localhost: If True, use 'localhost' instead of IP for display
        
    Returns:
        str: Formatted URL string
    """
    display_host = get_display_ip(host)
    
    if use_localhost and display_host in ("0.0.0.0", "127.0.0.1"):
        display_host = "localhost"
    
    return f"http://{display_host}:{port}"


# Example usage and test cases
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("0.0.0.0", "127.0.0.1" if platform.system().lower() == "windows" else "0.0.0.0"),
        ("127.0.0.1", "127.0.0.1"),
        ("192.168.1.1", "192.168.1.1"),
        ("localhost", "localhost"),
        ("", ValueError),  # Should raise ValueError
        (None, ValueError),  # Should raise ValueError
    ]
    
    print(f"Running on: {platform.system()}")
    print("-" * 50)
    
    for ip, expected in test_cases:
        try:
            result = get_display_ip(ip)
            if expected is ValueError:
                print(f"FAIL: {ip} should have raised ValueError")
            elif result == expected:
                print(f"PASS: {ip} -> {result}")
            else:
                print(f"FAIL: {ip} -> {result} (expected {expected})")
        except ValueError as e:
            if expected is ValueError:
                print(f"PASS: {ip} raised ValueError as expected")
            else:
                print(f"FAIL: {ip} raised unexpected ValueError: {e}")
        except Exception as e:
            print(f"ERROR: {ip} raised unexpected exception: {e}")
    
    print("-" * 50)
    
    # Test URL formatting
    print("\nURL Formatting Tests:")
    print(f"Server URL: {format_server_url('0.0.0.0', 8080)}")
    print(f"Server URL (no localhost): {format_server_url('0.0.0.0', 8080, use_localhost=False)}")
    print(f"Server URL: {format_server_url('127.0.0.1', 3000)}")
    print(f"Server URL: {format_server_url('192.168.1.100', 5000)}")
    
    # Test local IP detection
    local_ip = get_local_ip()
    if local_ip:
        print(f"\nLocal IP detected: {local_ip}")
        print(f"Display IP: {get_display_ip(local_ip)}")
    else:
        print("\nCould not detect local IP")