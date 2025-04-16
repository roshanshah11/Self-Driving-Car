import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from line_detection.limelight_manager import LimelightManager
from line_detection.track_detector import TrackDetector
from racing_line_optimization.racing_line_optimizer import RacingLineOptimizer
from racing_line_optimization.racing_controller import RacingController
from visualization.visualization import CameraVisualizer
from visualization.web_visualizer import WebVisualizer

class AutonomousRacer:
    def __init__(self):
        # Initialize components
        self.limelight = LimelightManager()
        self.track_detector = TrackDetector(map_size=(800, 800), cell_size=10)
        self.racing_optimizer = RacingLineOptimizer()
        self.controller = RacingController()
        self.visualizer = CameraVisualizer()
        self.web_visualizer = WebVisualizer(port=5000)
        self.web_visualizer.start()
        
        # Initialize variables
        self.optimal_racing_line = None
        self.speed_profile = None
        self.is_mapping_complete = False
        self.robot_pose = (400, 400, 0)  # Initial pose in map coordinates (x, y, theta)
        
        # Racing mode flag
        self.racing_mode = False
        
        # Track loop detection
        self.start_position = None
        self.position_history = []
        self.loop_detected = False
        self.min_track_points = 100  # Minimum points before considering loop detection
        self.min_distance_traveled = 200  # Minimum distance traveled before considering a loop
        self.loop_closure_threshold = 30  # Distance in pixels to consider a loop closure
        
    def setup(self):
        """Setup the system"""
        print("Setting up autonomous racing system...")
        
        # Connect to Limelight
        if not self.limelight.connect():
            print("Failed to connect to Limelight. Exiting.")
            return False
            
        # Configure Limelight pipeline for track detection
        self.limelight.setup_track_detection_pipeline()
        
        print("System setup complete!")
        return True
        
    def update_pose(self, imu_data=None):
        """
        Update robot pose using IMU data if available
        This is a simple implementation and should be replaced with proper odometry
        """
        if imu_data is not None:
            # Extract relevant IMU data
            # This is a placeholder - modify based on actual IMU data format
            delta_x = imu_data.get('delta_x', 0)
            delta_y = imu_data.get('delta_y', 0)
            delta_theta = imu_data.get('delta_theta', 0)
            
            # Update pose
            x, y, theta = self.robot_pose
            new_theta = theta + delta_theta
            
            # Move in the direction of the current heading
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            new_x = x + delta_x * cos_theta - delta_y * sin_theta
            new_y = y + delta_x * sin_theta + delta_y * cos_theta
            
            self.robot_pose = (new_x, new_y, new_theta)
        else:
            # Simple default movement when no IMU data is available
            x, y, theta = self.robot_pose
            # Move forward a small amount
            speed = 1.0
            x += speed * np.cos(theta)
            y += speed * np.sin(theta)
            self.robot_pose = (x, y, theta)
        
        # Record position history for loop detection
        self.position_history.append((self.robot_pose[0], self.robot_pose[1]))
        
        # Initialize start position if not set
        if self.start_position is None:
            self.start_position = (self.robot_pose[0], self.robot_pose[1])
    
    def map_track(self):
        """
        Map the track using camera data
        """
        # Get the latest image from Limelight
        image = self.limelight.get_latest_image()
        
        # Update raw image visualization
        self.visualizer.update_raw_image(image)
        self.web_visualizer.update_raw_image(image)
        
        # Detect track boundaries
        inner, outer = self.track_detector.detect_track_boundaries(image)
        
        # Update processed image visualization
        self.visualizer.update_processed_image(image, inner, outer)
        self.web_visualizer.update_processed_image(image, inner, outer)
        
        # If boundaries detected, update the map
        if inner is not None and outer is not None:
            # Update map with detected boundaries
            self.track_detector.update_map(inner, outer, self.robot_pose)
            
            # Visualize the map
            self.track_detector.visualize_track(
                show_optimal_line=self.racing_mode, 
                optimal_line=self.optimal_racing_line
            )
            
            # Update web visualization with track map
            self.web_visualizer.update_track_map(self.track_detector.get_visualization())
            
            return True
        else:
            return False
    
    def check_loop_closure(self):
        """
        Check if the robot has completed a full loop around the track
        Returns True if a loop is detected, False otherwise
        """
        if self.start_position is None or len(self.position_history) < self.min_track_points:
            return False
        
        # Calculate total distance traveled
        total_distance = 0
        for i in range(1, len(self.position_history)):
            p1 = self.position_history[i-1]
            p2 = self.position_history[i]
            total_distance += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Check if we've traveled enough distance
        if total_distance < self.min_distance_traveled:
            return False
        
        # Get current position and check distance to start position
        current_pos = self.position_history[-1]
        distance_to_start = np.sqrt(
            (current_pos[0] - self.start_position[0])**2 + 
            (current_pos[1] - self.start_position[1])**2
        )
        
        # Consider the recent path direction for more robust loop detection
        # Check if we're headed towards the start point
        if len(self.position_history) > 10:
            recent_vector = np.array([
                current_pos[0] - self.position_history[-10][0],
                current_pos[1] - self.position_history[-10][1]
            ])
            start_vector = np.array([
                self.start_position[0] - current_pos[0],
                self.start_position[1] - current_pos[1]
            ])
            
            # Normalize vectors
            recent_norm = np.linalg.norm(recent_vector)
            start_norm = np.linalg.norm(start_vector)
            
            if recent_norm > 0 and start_norm > 0:
                recent_unit = recent_vector / recent_norm
                start_unit = start_vector / start_norm
                
                # Calculate cosine similarity (dot product of unit vectors)
                similarity = np.dot(recent_unit, start_unit)
                
                # Increase threshold if we're moving toward the start
                heading_factor = max(0, similarity)
                adjusted_threshold = self.loop_closure_threshold * (1 + heading_factor)
            else:
                adjusted_threshold = self.loop_closure_threshold
        else:
            adjusted_threshold = self.loop_closure_threshold
        
        # Update web visualization with mapping progress
        self.web_visualizer.update_mapping_status(
            distance_to_start=distance_to_start,
            distance_traveled=total_distance,
            status=f"Mapping in progress... Distance to start: {distance_to_start:.2f}px"
        )
        
        # Check if we're close enough to the starting point
        if distance_to_start < adjusted_threshold:
            print(f"Loop detected! Distance to start: {distance_to_start:.2f} pixels")
            print(f"Total distance traveled: {total_distance:.2f} pixels")
            
            # Update web visualization with completion status
            self.web_visualizer.update_mapping_status(
                distance_to_start=distance_to_start,
                distance_traveled=total_distance,
                status="Track loop detected! Mapping complete.",
                mapping_complete=True
            )
            
            return True
            
        return False
            
    def find_optimal_racing_line(self):
        """
        Calculate the optimal racing line and speed profile after track mapping is complete
        """
        print("Calculating optimal racing line...")
        
        # Get the track map
        track_map = self.track_detector.track_map
        
        # Extract centerline as starting point
        centerline = self.racing_optimizer.extract_track_centerline(track_map)
        
        # Find the racing line - add progress output
        print("Step 1/3: Extracting track centerline...")
        if len(centerline) > 0:
            print(f"Centerline extracted with {len(centerline)} points")
            
            # Step 2: Optimize the racing line
            print("Step 2/3: Optimizing racing line based on track curvature...")
            self.optimal_racing_line = self.racing_optimizer.optimize_racing_line(track_map, centerline)
            
            if len(self.optimal_racing_line) > 0:
                # Step 3: Calculate speed profile
                print("Step 3/3: Calculating speed profile...")
                self.speed_profile = self.racing_optimizer.calculate_speed_profile(self.optimal_racing_line)
                
                # Set racing line in controller
                self.controller.set_racing_line(self.optimal_racing_line, self.speed_profile)
                
                # Set optimal racing line in track detector for visualization
                self.track_detector.optimal_line = self.optimal_racing_line
                
                # Calculate some statistics
                max_speed = max(self.speed_profile) if len(self.speed_profile) > 0 else 0
                avg_speed = np.mean(self.speed_profile) if len(self.speed_profile) > 0 else 0
                
                # Scale speeds to km/h for display (assuming speed profile is in m/s)
                max_speed_kmh = max_speed * 3.6
                avg_speed_kmh = avg_speed * 3.6
                
                # Print statistics
                print(f"Racing line calculated with {len(self.optimal_racing_line)} points")
                print(f"Maximum speed: {max_speed_kmh:.1f} km/h")
                print(f"Average speed: {avg_speed_kmh:.1f} km/h")
                
                # Visualize with racing line and speed profile
                self.track_detector.visualize_track(
                    show_optimal_line=True, 
                    optimal_line=self.optimal_racing_line,
                    speed_profile=self.speed_profile
                )
                
                # Update web visualization
                track_vis = self.track_detector.visualize_track(
                    show_optimal_line=True, 
                    optimal_line=self.optimal_racing_line, 
                    speed_profile=self.speed_profile
                )
                self.web_visualizer.update_track_map(track_vis)
                
                # Save racing line and speed profile to JSON file for future use
                try:
                    import json
                    import time
                    
                    # Create a timestamped filename
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"racing_line_data_{timestamp}.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    racing_data = {
                        'racing_line': self.optimal_racing_line.tolist(),
                        'speed_profile': self.speed_profile.tolist(),
                        'max_speed': float(max_speed_kmh),
                        'avg_speed': float(avg_speed_kmh),
                        'timestamp': timestamp
                    }
                    
                    with open(filename, 'w') as f:
                        json.dump(racing_data, f)
                    
                    print(f"Racing line data saved to {filename}")
                    
                    # Also save to the default filename for convenience
                    with open('racing_line_data.json', 'w') as f:
                        json.dump(racing_data, f)
                    
                except Exception as e:
                    print(f"Warning: Could not save racing line data to file: {e}")
                
                print("Optimal racing line calculation complete!")
                return True
            else:
                print("Failed to optimize racing line")
                return False
        else:
            print("Failed to extract centerline from track map")
            return False
    
    def run_mapping_mode(self, max_duration=600):
        """
        Run mapping mode until a loop is detected or max_duration is reached
        max_duration: Maximum mapping time in seconds (default 10 minutes)
        """
        print("Starting track mapping...")
        print("Drive the car around the track once. Mapping will continue until a full loop is detected.")
        print("Press Ctrl+C to stop mapping manually.")
        print("Camera visualizations are being displayed in separate windows.")
        print(f"Web visualization available at http://localhost:5000 or your local IP address")
        
        self.start_position = None
        self.position_history = []
        self.loop_detected = False
        
        start_time = time.time()
        last_status_time = start_time
        status_interval = 5  # Show status every 5 seconds
        
        try:
            while time.time() - start_time < max_duration:
                current_time = time.time()
                
                # Get IMU data for pose update
                imu_data = self.limelight.get_imu_data()
                
                # Update robot pose (this will also update position history)
                self.update_pose(imu_data)
                
                # Update track map
                self.map_track()
                
                # Check for loop closure
                if self.check_loop_closure():
                    self.loop_detected = True
                    break
                    
                # Show periodic status
                if current_time - last_status_time > status_interval:
                    elapsed = current_time - start_time
                    print(f"Mapping in progress... ({elapsed:.1f}s elapsed)")
                    if len(self.position_history) > self.min_track_points:
                        # Calculate distance to starting position
                        current_pos = self.position_history[-1]
                        distance_to_start = np.sqrt(
                            (current_pos[0] - self.start_position[0])**2 + 
                            (current_pos[1] - self.start_position[1])**2
                        )
                        print(f"Distance to start position: {distance_to_start:.2f} pixels")
                    last_status_time = current_time
                
                # Sleep to control update rate
                time.sleep(0.05)
            
            if self.loop_detected:
                print("Full track loop detected! Mapping complete.")
            else:
                print("Maximum mapping duration reached without detecting a complete loop.")
                
            self.is_mapping_complete = True
            
            # Calculate optimal racing line
            self.find_optimal_racing_line()
            
        except KeyboardInterrupt:
            print("\nMapping interrupted by user.")
            if len(self.position_history) > self.min_track_points:
                print("Sufficient data collected. Proceeding with available map.")
                self.is_mapping_complete = True
                self.find_optimal_racing_line()
            else:
                print("Insufficient data collected. Please try mapping again.")
            
    def run_racing_mode(self, duration=300):
        """
        Run racing mode following the optimal racing line
        """
        if not self.is_mapping_complete or self.optimal_racing_line is None:
            print("Cannot start racing mode: mapping not complete or racing line not available")
            return
            
        print("Starting racing mode!")
        self.racing_mode = True
        
        start_time = time.time()
        last_update_time = start_time
        
        try:
            while time.time() - start_time < duration:
                current_time = time.time()
                dt = current_time - last_update_time
                last_update_time = current_time
                
                # Get IMU data for pose update
                imu_data = self.limelight.get_imu_data()
                
                # Update robot pose
                self.update_pose(imu_data)
                
                # Get control signals
                throttle, steering = self.controller.update_control(
                    self.robot_pose, 
                    self.robot_pose[2],  # heading
                    dt
                )
                
                # Apply controls to robot
                print(f"Throttle: {throttle:.2f}, Steering: {steering:.2f}")
                
                # Update map and visualization
                self.map_track()
                
                # Sleep to control update rate
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("Racing interrupted!")
        finally:
            self.racing_mode = False
    
    def run(self):
        """
        Main run function
        """
        if not self.setup():
            return
        
        try:
            # First run mapping mode until a loop is detected or user interrupts
            self.run_mapping_mode()
            
            # Then run racing mode if mapping was successful
            if self.is_mapping_complete and self.optimal_racing_line is not None:
                user_input = input("Mapping complete. Start autonomous racing mode? (y/n): ")
                if user_input.lower() == 'y':
                    self.run_racing_mode()
                
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            # Cleanup
            self.limelight.disconnect()
            self.visualizer.cleanup()
            plt.close('all')
            print("Program terminated")

if __name__ == "__main__":
    racer = AutonomousRacer()
    racer.run()