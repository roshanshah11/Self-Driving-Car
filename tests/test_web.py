import requests

def test_web_interface():
    """Test accessing the web interface pages"""
    base_url = "http://localhost:8080"
    
    # Test index page
    print("Testing index page...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status code: {response.status_code}")
        if response.status_code != 200:
            print(f"Error: {response.text}")
        else:
            print("Success!")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test API endpoints
    endpoints = [
        "/status", 
        "/api/track_data",
        "/video_feed",
        "/video_processed",
        "/track_map"
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}...")
        try:
            response = requests.get(f"{base_url}{endpoint}")
            print(f"Status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Error: {response.text}")
            else:
                print("Success!")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_web_interface() 