import numpy as np
import cv2
import time
import json

# Import from the centralized limelight connection module
from line_detection.limelight_connection import (
    connect_to_limelight, 
    set_camera_params, 
    switch_pipeline, 
    get_line_data, 
    disconnect,
    LIMELIGHT_AVAILABLE
)

class LimelightManager:
    def __init__(self):
        self.ll = None
        self.limelight_address = None
        self.nt_client = None
        self.connected = False
        
        # Detection parameters
        self.detection_params = {
            'threshold_min': 200,  # Higher threshold for white pixel detection (was 180)
            'exposure': 1500.0,    # Lower exposure for better white line contrast (was 10.0)
            'brightness': 60,      # Slightly lower brightness for better contrast (was 75)
            'gain': 50             # Camera gain (0-100)
        }
        
        # Store latest results
        self.latest_results = None
        self.latest_parsed_results = None
        self.latest_status = None
        
        # Store latest line positions
        self.latest_left_line = None
        self.latest_right_line = None
        self.latest_track_center = None
        
    def connect(self):
        """Connect to the first available Limelight camera using the centralized utility"""
        if not LIMELIGHT_AVAILABLE:
            print("ERROR: Limelight libraries not available. Please install them with:")
            print("pip install limelight-vision")
            return False
        
        try:
            # Create configuration dict with our detection parameters
            config = {
                'exposure': self.detection_params['exposure'],
                'brightness': self.detection_params['brightness'],
                'gain': self.detection_params['gain'],
                'threshold_min': self.detection_params['threshold_min'],
                'enable_websocket': True
            }
            
            # Use the centralized connection utility
            self.ll, self.limelight_address, self.nt_client = connect_to_limelight(
                config=config, 
                use_nt=True, 
                debug=True
            )
            
            if self.ll is None:
                print("ERROR: Failed to connect to Limelight")
                return False
                
            print(f"Connected to Limelight at {self.limelight_address}")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"Error connecting to Limelight: {e}")
            print("Please verify your Limelight camera is properly connected")
            return False
    
    def set_camera_exposure(self, exposure_ms):
        """Set the camera exposure in milliseconds using the centralized utility"""
        if not self.connected:
            return False
            
        try:
            # Use the centralized utility function
            success = set_camera_params(
                ll=self.ll,
                nt_client=self.nt_client,
                exposure=exposure_ms
            )
            
            if success:
                # Update our local parameter
                self.detection_params['exposure'] = exposure_ms
                print(f"Camera exposure set to {exposure_ms} ms")
                return True
            else:
                print("Failed to set camera exposure")
                return False
                
        except Exception as e:
            print(f"Failed to set camera exposure: {e}")
            return False
    
    def setup_track_detection_pipeline(self):
        """Configure Limelight pipeline for white track detection"""
        # Create a pipeline optimized for white line detection using Limelight's capabilities
        pipeline_config = {
            # Color thresholding for white tape - wider range for better detection
            'hue_low': 0,
            'hue_high': 180,  # Full hue range
            'sat_low': 0,
            'sat_high': 50,   # Increased to catch off-white colors
            'val_low': 200,   # Higher threshold for white detection (was 180)
            'val_high': 255,
            
            # Enable HSV filtering
            'hsv_filter': 1,
            
            # Contour filtering parameters
            'area_min': 50,  # Minimum contour area to be considered
            'area_max': 100000,
            'solidity_min': 0, # No minimum solidity constraint
            'solidity_max': 100,
            
            # Enable contour processing
            'contour_mode': 1,  # External contours
            'contour_algorithm': 1,  # Simple approximation
            
            # Turn on desaturation to improve white detection
            'desaturate': 1,
            
            # Enable edge detection
            'edge_detect': 1,
            
            # Camera settings - these will be set using the centralized utility
            'exposure': self.detection_params['exposure'],
            'brightness': self.detection_params['brightness'],
            'gain': self.detection_params['gain'],
            
            # Enable Cross-hair in results
            'cross_hair_x': 1.0,
            'cross_hair_y': 1.0,
            
            # Enable dual threshold and erosion/dilation for better noise handling
            'dual_threshold': 1,
            'erosion': 1,
            'dilation': 1,
            
            # Corner detection settings for track detection
            'corner_detect': 1,
            'corner_threshold': 30
        }
        
        # Update the pipeline and save it
        try:
            # First set the camera parameters using the centralized utility
            set_camera_params(
                ll=self.ll,
                nt_client=self.nt_client,
                exposure=self.detection_params['exposure'],
                brightness=self.detection_params['brightness'],
                gain=self.detection_params['gain']
            )
            
            # Then update the rest of the pipeline config directly
            self.ll.update_pipeline(json.dumps(pipeline_config), flush=1)
            print("Track detection pipeline configured with white line detection optimizations")
            
            # Verify the changes
            updated_pipeline = self.ll.get_pipeline_atindex(0)
            print(f"Updated pipeline settings: {updated_pipeline}")
            
            return True
        except Exception as e:
            print(f"Error configuring pipeline: {e}")
            return False
    
    def update_detection_params(self, params):
        """Update detection parameters and apply to Limelight"""
        # Update local parameters
        if 'threshold_min' in params:
            self.detection_params['threshold_min'] = params['threshold_min']
        if 'exposure' in params:
            self.detection_params['exposure'] = params['exposure']
        if 'brightness' in params:
            self.detection_params['brightness'] = params['brightness']
        if 'gain' in params and 'gain' in self.detection_params:
            self.detection_params['gain'] = params['gain']
            
        print(f"Updating detection parameters: {self.detection_params}")
            
        # Apply to Limelight using the centralized utility
        if self.connected and self.ll is not None:
            try:
                # Update camera settings with the centralized utility
                success = set_camera_params(
                    ll=self.ll,
                    nt_client=self.nt_client,
                    exposure=self.detection_params['exposure'],
                    brightness=self.detection_params['brightness'],
                    gain=self.detection_params['gain']
                )
                
                # Update threshold (val_low) separately since it's not a camera parameter
                pipeline_update = {
                    'val_low': self.detection_params['threshold_min']
                }
                
                # Use direct API for parameters not in centralized utility
                self.ll.update_pipeline(json.dumps(pipeline_update), flush=1)
                
                # Verify the changes
                return True
            except Exception as e:
                print(f"Error updating Limelight parameters: {e}")
                return False
        return False
        
    def get_latest_image(self):
        """Get the latest processed image from Limelight"""
        if not self.connected:
            return None
            
        # Get data from Limelight
        try:
            # Try to get line data using centralized utility
            left_line, right_line = get_line_data(self.ll, self.nt_client)
            
            # Create a visualization of the detected lines
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw the lines on the image
            if left_line is not None:
                x1, y1, x2, y2 = [int(c) for c in left_line]
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue for left line
                cv2.putText(img, 'Left Line', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
            if right_line is not None:
                x1, y1, x2, y2 = [int(c) for c in right_line]
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red for right line
                cv2.putText(img, 'Right Line', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # If we have both lines, fill the area between them with green
            if left_line is not None and right_line is not None:
                # Create a polygon for the fill area
                points = np.array([
                    [int(left_line[0]), int(left_line[1])],   # Left start
                    [int(left_line[2]), int(left_line[3])],   # Left end
                    [int(right_line[2]), int(right_line[3])], # Right end
                    [int(right_line[0]), int(right_line[1])]  # Right start
                ], dtype=np.int32)
                
                # Fill with semi-transparent green
                overlay = img.copy()
                cv2.fillPoly(overlay, [points], (0, 255, 0))  # Pure green
                cv2.addWeighted(overlay, 0.5, img, 1, 0, img)  # Apply with transparency
            
            # Add title and information
            cv2.putText(img, 'Limelight Camera Feed', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # If we have at least one line, store and return the visualization
            if left_line is not None or right_line is not None:
                # Store the latest lines
                self.latest_left_line = left_line
                self.latest_right_line = right_line
                return img
                
            # If no lines detected, return a blank image with text
            cv2.putText(img, "No track lines detected", (10, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return img
            
        except Exception as e:
            print(f"Failed to get image from Limelight: {e}")
            # Return a blank image with error text
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Limelight error: {e}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            return error_img
    
    def get_track_boundaries(self):
        """
        Get the detected track boundaries from Limelight
        Returns inner_boundary, outer_boundary
        """
        if not self.connected:
            return None, None
            
        # Get latest line data
        try:
            left_line, right_line = get_line_data(self.ll, self.nt_client)
            
            # Convert lines to contour format
            inner_boundary = None
            outer_boundary = None
            
            if left_line is not None:
                # Convert left line to inner boundary contour
                x1, y1, x2, y2 = [int(c) for c in left_line]
                inner_boundary = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32)
            
            if right_line is not None:
                # Convert right line to outer boundary contour
                x1, y1, x2, y2 = [int(c) for c in right_line]
                outer_boundary = np.array([[[x1, y1]], [[x2, y2]]], dtype=np.int32)
            
            return inner_boundary, outer_boundary
                
        except Exception as e:
            print(f"Failed to get track boundaries: {e}")
            return None, None
    
    def get_latest_contours(self):
        """Get the latest contours detected by Limelight"""
        inner_boundary, outer_boundary = self.get_track_boundaries()
        
        # Convert to list of contours
        contours = []
        if inner_boundary is not None:
            contours.append(inner_boundary)
        if outer_boundary is not None:
            contours.append(outer_boundary)
            
        return contours if contours else None
            
    def get_imu_data(self):
        """Get IMU data from Limelight if available"""
        if not self.connected:
            return None
            
        # Get IMU data from the real Limelight
        try:
            status = self.ll.get_status()
            # Extract IMU data if available in status
            if 'imu' in status:
                return status['imu']
            return None
        except Exception as e:
            print(f"Failed to get IMU data: {e}")
            return None
        
    def disconnect(self):
        """Disconnect from Limelight using the centralized utility"""
        if self.connected and self.ll is not None:
            try:
                # Use the centralized utility
                disconnect(self.ll, self.nt_client)
                print("Disconnected from Limelight")
            except Exception as e:
                print(f"Error disconnecting from Limelight: {e}")
            finally:
                self.connected = False