import numpy as np
import cv2
from racing_line_optimizer import RacingLineOptimizer
import json
import time

def create_track():
    """Create a more interesting track with varying width and curvature"""
    print("Creating an interesting track with varying width and curvature...")
    
    # Create a blank map
    track_map = np.zeros((800, 800), dtype=np.uint8)
    vis_map = np.zeros((800, 800, 3), dtype=np.uint8)
    
    # Draw gridlines
    for x in range(0, 800, 100):
        cv2.line(vis_map, (x, 0), (x, 799), (30, 30, 30), 1)
    for y in range(0, 800, 100):
        cv2.line(vis_map, (0, y), (799, y), (30, 30, 30), 1)
    
    # Track parameters - figure-8 with varying width
    center_x, center_y = 400, 400
    
    # Draw track using multiple ellipses with varying widths
    # First, create two overlapping ellipses for a figure-8 shape
    mask = np.zeros((800, 800), dtype=np.uint8)
    
    # Top ellipse - wider
    top_width = 120
    cv2.ellipse(mask, (center_x, center_y - 100), (200, 150), 
               0, 0, 360, 255, -1)
    cv2.ellipse(mask, (center_x, center_y - 100), (200 - top_width, 150 - top_width), 
               0, 0, 360, 0, -1)
    
    # Bottom ellipse - narrower 
    bottom_width = 80
    cv2.ellipse(mask, (center_x, center_y + 100), (180, 130), 
               0, 0, 360, 255, -1)
    cv2.ellipse(mask, (center_x, center_y + 100), (180 - bottom_width, 130 - bottom_width), 
               0, 0, 360, 0, -1)
    
    # Add a chicane section on the right side
    chicane_points = np.array([
        [580, 350],
        [620, 380],
        [600, 420],
        [640, 450],
        [600, 480]
    ], dtype=np.int32)
    
    # Draw the chicane with a varying width
    cv2.polylines(mask, [chicane_points], False, 255, 60)
    
    # Convert to binary map (1 = track)
    track_map = (mask > 0).astype(np.uint8)
    
    # Create visualization
    vis_map[track_map == 1] = (0, 200, 0)  # Green for track
    
    # Save visualization
    cv2.imwrite("interesting_track_map.jpg", vis_map)
    print("Created interesting track and saved to interesting_track_map.jpg")
    
    return track_map, vis_map

def generate_racing_line(track_map, vis_map):
    """Generate an optimized racing line with realistic speed profile"""
    print("\nStarting racing line optimization...")
    start_time = time.time()
    
    # Create racing line optimizer with more realistic car params
    car_params = {
        'max_velocity': 8.0,  # m/s (about 29 km/h)
        'max_acceleration': 3.0,  # m/s^2
        'max_deceleration': 6.0,  # m/s^2 (stronger braking than acceleration)
        'min_turn_radius': 0.5,  # meters
        'wheelbase': 0.3,  # meters
        'mass': 3.0,  # kg
    }
    
    optimizer = RacingLineOptimizer(car_params=car_params)
    
    # Extract centerline
    print("Extracting track centerline...")
    centerline = optimizer.extract_track_centerline(track_map)
    print(f"Extracted centerline with {len(centerline)} points")
    
    # Draw centerline on visualization
    centerline_vis = vis_map.copy()
    for point in centerline:
        x, y = int(point[0]), int(point[1])
        cv2.circle(centerline_vis, (x, y), 2, (0, 0, 255), -1)
    cv2.imwrite("interesting_centerline.jpg", centerline_vis)
    
    # Generate optimal racing line
    print("Optimizing racing line using PSO algorithm...")
    print("This may take a few minutes...")
    racing_line = optimizer.optimize_racing_line(track_map, centerline)
    print(f"Generated racing line with {len(racing_line)} points")
    print(f"Optimization took {time.time() - start_time:.1f} seconds")
    
    # Convert racing line to integer array for drawing
    racing_line_int = np.array(racing_line, dtype=np.int32)
    
    # Draw racing line on visualization
    racing_line_vis = vis_map.copy()
    cv2.polylines(racing_line_vis, [racing_line_int], True, (255, 165, 0), 3)
    cv2.imwrite("interesting_racing_line.jpg", racing_line_vis)
    
    # Calculate speed profile with realistic constraints
    print("\nCalculating speed profile with realistic vehicle constraints...")
    speed_profile = optimizer.calculate_speed_profile(racing_line)
    
    # Get speed statistics
    max_speed = np.max(speed_profile)
    min_speed = np.min(speed_profile)
    avg_speed = np.mean(speed_profile)
    print(f"Speed profile statistics:")
    print(f"  Min speed: {min_speed:.2f} m/s ({min_speed*3.6:.2f} km/h)")
    print(f"  Max speed: {max_speed:.2f} m/s ({max_speed*3.6:.2f} km/h)")
    print(f"  Avg speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.2f} km/h)")
    
    # Create a visualization with the speed profile
    speed_vis = vis_map.copy()
    
    # Draw racing line with color based on speed
    for i in range(len(racing_line_int) - 1):
        # Ensure we don't go out of bounds with speed profile
        if i >= len(speed_profile):
            break
            
        # Normalize speed for this segment
        t = (speed_profile[i] - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.5
        
        # Color based on speed: blue (slow) to red (fast)
        b = int(255 * (1 - t))
        r = int(255 * t)
        g = 0
        
        # Draw segment with color based on speed - ensure points are tuples
        pt1 = (int(racing_line_int[i][0]), int(racing_line_int[i][1]))
        pt2 = (int(racing_line_int[i+1][0]), int(racing_line_int[i+1][1]))
        cv2.line(speed_vis, pt1, pt2, (b, g, r), 3)
    
    # Add markers with speed values at regular intervals
    num_markers = min(16, len(racing_line_int))  # Show more markers for an interesting track
    marker_indices = np.linspace(0, len(racing_line_int)-1, num_markers, dtype=int)
    
    for idx in marker_indices:
        # Ensure we don't go out of bounds with speed profile
        if idx >= len(speed_profile):
            continue
            
        # Get marker position and speed
        x = int(racing_line_int[idx][0])
        y = int(racing_line_int[idx][1])
        marker_pos = (x, y)
        speed_val = speed_profile[idx]
        
        # Color based on speed
        t = (speed_val - min_speed) / (max_speed - min_speed) if max_speed > min_speed else 0.5
        b = int(255 * (1 - t))
        r = int(255 * t)
        g = 0
        
        # Draw marker
        cv2.circle(speed_vis, marker_pos, 6, (b, g, r), -1)
        cv2.circle(speed_vis, marker_pos, 6, (255, 255, 255), 1)  # White outline
        
        # Show speed in km/h
        speed_km_h = speed_val * 3.6
        cv2.putText(speed_vis, f"{speed_km_h:.1f}", 
                   (marker_pos[0] + 8, marker_pos[1] + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Add color gradient legend
    legend_width = 150
    legend_height = 15
    legend_x = 20
    legend_y = 40
    
    cv2.putText(speed_vis, "Speed Profile (km/h):", (legend_x, legend_y - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw gradient bar
    for i in range(legend_width):
        t = i / legend_width
        b = int(255 * (1 - t))
        r = int(255 * t)
        g = 0
        cv2.line(speed_vis, (legend_x + i, legend_y), 
                (legend_x + i, legend_y + legend_height), (b, g, r), 1)
    
    # Add labels
    min_speed_kmh = min_speed * 3.6
    max_speed_kmh = max_speed * 3.6
    cv2.putText(speed_vis, f"{min_speed_kmh:.1f}", (legend_x - 5, legend_y + legend_height + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 255), 1)
    cv2.putText(speed_vis, f"{max_speed_kmh:.1f}", (legend_x + legend_width - 5, legend_y + legend_height + 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 150), 1)
    
    # Save visualization with speed profile
    cv2.imwrite("interesting_speed_profile.jpg", speed_vis)
    print("Created speed profile visualization and saved to interesting_speed_profile.jpg")
    
    # Save racing line data
    racing_data = {
        'racing_line': racing_line.tolist(),
        'speed_profile': speed_profile.tolist(),
        'max_speed': float(max_speed),
        'min_speed': float(min_speed),
        'avg_speed': float(avg_speed)
    }
    
    with open('interesting_racing_line_data.json', 'w') as f:
        json.dump(racing_data, f)
    print("Saved racing line data to interesting_racing_line_data.json")
    
    # Update the fixed_racing_line_data.json file for the standard visualization
    with open('fixed_racing_line_data.json', 'w') as f:
        json.dump(racing_data, f)
    print("Also updated fixed_racing_line_data.json")
    
    # Also generate the latest_track_map.jpg using the update_vis.py logic
    try:
        # Create a TrackDetector instance
        from track_detector import TrackDetector
        track = TrackDetector()
        
        # Set the track map
        track.track_map = track_map
        
        # Set the racing line
        track.optimal_line = racing_line_int
        
        # Generate visualization with racing line and speed profile
        track_vis = track.visualize_track(show_optimal_line=True, 
                                         optimal_line=racing_line_int, 
                                         speed_profile=speed_profile)
        
        # Save visualization
        cv2.imwrite('latest_track_map.jpg', track_vis)
        print('Updated latest_track_map.jpg with new racing line and speed profile')
    except Exception as e:
        print(f"Error updating latest_track_map.jpg: {e}")
    
    return racing_line, speed_profile

if __name__ == "__main__":
    print("Generating interesting track with varying speeds...")
    track_map, vis_map = create_track()
    racing_line, speed_profile = generate_racing_line(track_map, vis_map)
    print("\nAll done! Check the generated images for results:")
    print("1. interesting_track_map.jpg - The track layout")
    print("2. interesting_centerline.jpg - The track centerline")
    print("3. interesting_racing_line.jpg - The optimized racing line")
    print("4. interesting_speed_profile.jpg - The racing line with speed indicators")
    print("5. latest_track_map.jpg - Updated track map for the standard visualization")