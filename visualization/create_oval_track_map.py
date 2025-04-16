import numpy as np
import cv2
from racing_line_optimizer import RacingLineOptimizer
import json

# Create a proper oval track binary map
def create_oval_track():
    # Create a blank map
    track_map = np.zeros((800, 800), dtype=np.uint8)
    
    # Track parameters
    center_x, center_y = 400, 400
    a, b = 250, 150  # Horizontal/vertical radii
    track_width = 100
    
    # Create masks for inner and outer ellipses
    mask = np.zeros((800, 800), dtype=np.uint8)
    
    # Draw outer ellipse
    cv2.ellipse(mask, (center_x, center_y), (a + track_width//2, b + track_width//2), 
                0, 0, 360, 255, -1)
    
    # Draw inner ellipse (hole)
    cv2.ellipse(mask, (center_x, center_y), (a - track_width//2, b - track_width//2), 
                0, 0, 360, 0, -1)
    
    # Convert to binary map (1 = track)
    track_map = (mask > 0).astype(np.uint8)
    
    # Create visualization
    vis_map = np.zeros((800, 800, 3), dtype=np.uint8)
    vis_map[track_map == 1] = (0, 200, 0)  # Green for track
    
    # Draw gridlines
    for x in range(0, 800, 100):
        cv2.line(vis_map, (x, 0), (x, 799), (30, 30, 30), 1)
    for y in range(0, 800, 100):
        cv2.line(vis_map, (0, y), (799, y), (30, 30, 30), 1)
    
    # Add coordinates
    for x in range(0, 800, 100):
        for y in range(0, 800, 100):
            cv2.putText(vis_map, f"{x},{y}", (x+5, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
    
    # Save visualization
    cv2.imwrite("fixed_track_map.jpg", vis_map)
    
    return track_map, vis_map

# Generate racing line
def generate_racing_line(track_map, vis_map):
    # Create racing line optimizer
    optimizer = RacingLineOptimizer()
    
    # Extract centerline
    print("Extracting centerline...")
    centerline = optimizer.extract_track_centerline(track_map)
    print(f"Extracted centerline with {len(centerline)} points")
    
    # Draw centerline on visualization
    centerline_vis = vis_map.copy()
    for point in centerline:
        x, y = int(point[0]), int(point[1])
        cv2.circle(centerline_vis, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite("fixed_centerline.jpg", centerline_vis)
    
    # Generate optimal racing line
    print("Optimizing racing line...")
    racing_line = optimizer.optimize_racing_line(track_map, centerline)
    print(f"Generated racing line with {len(racing_line)} points")
    
    # Draw racing line on visualization
    racing_line_vis = vis_map.copy()
    racing_line_arr = np.array(racing_line, dtype=np.int32)
    cv2.polylines(racing_line_vis, [racing_line_arr], False, (255, 165, 0), 3)
    cv2.imwrite("fixed_racing_line.jpg", racing_line_vis)
    
    # Calculate speed profile
    print("Calculating speed profile...")
    speed_profile = optimizer.calculate_speed_profile(racing_line)
    
    # Save racing line data
    racing_data = {
        'racing_line': racing_line.tolist(),
        'speed_profile': speed_profile.tolist(),
        'max_speed': 15.0
    }
    
    with open('fixed_racing_line_data.json', 'w') as f:
        json.dump(racing_data, f)
    print("Saved racing line data to fixed_racing_line_data.json")
    
    return racing_line, speed_profile

# Run the functions
if __name__ == "__main__":
    track_map, vis_map = create_oval_track()
    racing_line, speed_profile = generate_racing_line(track_map, vis_map)
