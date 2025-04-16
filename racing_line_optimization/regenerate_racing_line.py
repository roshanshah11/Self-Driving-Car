import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from racing_line_optimizer import RacingLineOptimizer

def load_track_map():
    """Load track map from latest_track_map.jpg"""
    print("Loading track map from latest_track_map.jpg...")
    
    # Load the image
    img = cv2.imread("latest_track_map.jpg")
    if img is None:
        print("Error: Could not load latest_track_map.jpg")
        return None
    
    # Convert RGB to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get the track
    _, track_map = cv2.threshold(gray, 30, 1, cv2.THRESH_BINARY)
    
    print(f"Loaded track map with shape {track_map.shape}")
    return track_map

def generate_racing_line(track_map):
    """Generate an optimized racing line"""
    print("\nStarting racing line optimization...")
    
    # Create optimizer with reasonable car parameters
    car_params = {
        'max_velocity': 8.0,  # m/s (about 29 km/h)
        'max_acceleration': 3.0,  # m/s^2
        'max_deceleration': 6.0,  # m/s^2
        'min_turn_radius': 0.5,  # meters
        'wheelbase': 0.3,  # meters
        'mass': 3.0,  # kg
    }
    
    optimizer = RacingLineOptimizer(car_params=car_params)
    
    # Extract centerline
    print("Extracting track centerline...")
    centerline = optimizer.extract_track_centerline(track_map)
    print(f"Extracted centerline with {len(centerline)} points")
    
    # Generate optimal racing line
    print("Optimizing racing line using PSO algorithm...")
    racing_line = optimizer.optimize_racing_line(track_map, centerline)
    
    # Ensure racing line is an integer numpy array for visualization
    racing_line_int = np.array(racing_line, dtype=np.int32)
    print(f"Generated racing line with {len(racing_line_int)} points")
    
    # Calculate speed profile
    print("\nCalculating speed profile...")
    speed_profile = optimizer.calculate_speed_profile(racing_line)
    
    # Save racing line data to JSON
    racing_data = {
        'racing_line': racing_line.tolist(),
        'speed_profile': speed_profile.tolist(),
        'max_speed': 15.0
    }
    
    with open('racing_line_data.json', 'w') as f:
        json.dump(racing_data, f)
    print("Saved racing line data to racing_line_data.json")
    
    return racing_line_int, speed_profile

def visualize_racing_line(track_map, racing_line, speed_profile=None):
    """Create a visualization with the racing line"""
    # Create a color visualization
    vis_map = cv2.cvtColor(track_map * 100, cv2.COLOR_GRAY2BGR)
    
    # Draw the racing line
    if speed_profile is not None:
        # Color by speed
        max_speed = np.max(speed_profile)
        min_speed = np.min(speed_profile)
        
        for i in range(len(racing_line) - 1):
            # Normalize speed for color
            t = (speed_profile[i] - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.5
            
            # Blue (slow) to red (fast)
            b = int(255 * (1 - t))
            r = int(255 * t)
            g = 0
            
            # Ensure points are correctly formatted as tuples
            pt1 = (int(racing_line[i][0]), int(racing_line[i][1]))
            pt2 = (int(racing_line[i+1][0]), int(racing_line[i+1][1]))
            
            # Draw the line segment
            try:
                cv2.line(vis_map, pt1, pt2, (b, g, r), 3)
            except Exception as e:
                print(f"Error drawing line: {e}")
                print(f"Points: {pt1}, {pt2}")
    else:
        # Single color (orange)
        for i in range(len(racing_line) - 1):
            # Ensure points are correctly formatted
            pt1 = (int(racing_line[i][0]), int(racing_line[i][1]))
            pt2 = (int(racing_line[i+1][0]), int(racing_line[i+1][1]))
            
            try:
                cv2.line(vis_map, pt1, pt2, (0, 165, 255), 3)
            except Exception as e:
                print(f"Error drawing line: {e}")
                print(f"Points: {pt1}, {pt2}")
    
    # Save visualization
    cv2.imwrite("regenerated_racing_line.jpg", vis_map)
    print("Saved visualization to regenerated_racing_line.jpg")
    
    return vis_map

def main():
    # Load track map
    track_map = load_track_map()
    if track_map is None:
        return
    
    # Generate racing line
    racing_line, speed_profile = generate_racing_line(track_map)
    
    # Visualize
    visualize_racing_line(track_map, racing_line, speed_profile)
    
    print("\nProcess complete! Check regenerated_racing_line.jpg for results.")

if __name__ == "__main__":
    main() 