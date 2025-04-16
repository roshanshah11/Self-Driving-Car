import os
import json
import time
import numpy as np
import threading
import logging
import cv2

# Import limelight modules with centralized connection utility
import limelight
from line_detection import limelightresults
from line_detection.limelight_connection import connect_to_limelight, get_line_data, set_camera_params, switch_pipeline, disconnect
from line_detection.detect_limelight import get_latest_image
from visualization.web_visualizer import WebVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("racer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LineDetector")

class Config:
    """Configuration for the line detection system"""
    
    # Camera and detection settings
    CAMERA = {
        'exposure': 1500.0,      # Lower exposure for better white line contrast
        'brightness': 60,        # Brightness for better contrast
        'gain': 50,              # Camera gain
        'pipeline': 1,           # Using pipeline 1 for line detection
        'threshold_min': 200,    # Threshold for white detection
    }
    
    # Network settings
    NETWORK = {
        'web_port': 8080,
        'python_config_endpoint': 'http://{address}:5801/pythonconfig',
        'max_retries': 3,
        'retry_delay': 1,
        'request_timeout': 2,
    }
    
    # Threading and timing
    THREADING = {
        'data_polling_rate': 0.02,        # How frequently to poll for data (in seconds)
        'screen_update_interval': 0.05,   # How frequently to update the screen (in seconds)
        'max_consecutive_errors': 5,      # Maximum consecutive errors before logging
        'error_log_cooldown': 20,         # How many iterations to wait before logging errors again
    }

class LineDetector:
    def __init__(self):
        # Initialize components
        self.limelight_address = None
        self.ll = None
        
        # For storing line detection data
        self.latest_left_line = None
        self.latest_right_line = None
        self.python_data_lock = threading.Lock()
        
        # Initialize web visualizer
        self.web_visualizer = WebVisualizer(
            limelight_manager=None,
            port=Config.NETWORK['web_port']
        )
        
        # NetworkTables client for getting data from Limelight
        self.nt_client = None
        
    def setup(self):
        """Setup the system"""
        logger.info("Setting up white line detection system...")
        
        try:
            # Connect to Limelight using the centralized connection utility
            config = {
                'exposure': Config.CAMERA['exposure'],
                'brightness': Config.CAMERA['brightness'],
                'gain': Config.CAMERA['gain'],
                'pipeline': Config.CAMERA['pipeline'],
                'threshold_min': Config.CAMERA['threshold_min'],
                'request_timeout': Config.NETWORK['request_timeout']
            }
            
            self.ll, self.limelight_address, self.nt_client = connect_to_limelight(config=config, use_nt=True, debug=True)
            
            if self.ll is None:
                logger.error("Failed to connect to any Limelight cameras. Cannot continue.")
                return False
                
            logger.info(f"Successfully connected to Limelight at {self.limelight_address}")
            
            # Connect Limelight to web visualizer
            self.web_visualizer.set_limelight_address(self.limelight_address)
            self.web_visualizer.set_limelight_instance(self.ll)
            
            # Set up a parameter update callback
            self.web_visualizer.set_param_update_callback(self.update_detection_params)
            
            # Configure Python-specific parameters
            self.configure_python_parameters()
            
            # Start the data fetching thread
            self.start_data_fetching_thread()
            
            # Start the web visualizer
            self.web_visualizer.start()
            
            logger.info("System setup complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
            return False
            
    def configure_python_parameters(self):
        """Configure Python-specific parameters for the vision pipeline"""
        try:
            # Add retry logic for HTTP requests
            max_retries = Config.NETWORK['max_retries']
            retry_delay = Config.NETWORK['retry_delay']
            python_params = {
                'threshold_min': Config.CAMERA['threshold_min']
            }
            
            for attempt in range(max_retries):
                try:
                    # Send to Limelight Python config endpoint
                    endpoint = Config.NETWORK['python_config_endpoint'].format(address=self.limelight_address)
                    import requests
                    response = requests.post(
                        endpoint,
                        json=python_params,
                        timeout=Config.NETWORK['request_timeout']
                    )
                    if response.status_code == 200:
                        logger.info("Python parameters set successfully")
                        return True
                    else:
                        logger.warning(f"Failed to set Python parameters: {response.status_code}")
                except Exception as e:
                    logger.error(f"Error setting Python parameters (attempt {attempt+1}/{max_retries}): {e}")
                    
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
            
            logger.error("Failed to set Python parameters after all retries")
            return False
        except Exception as e:
            logger.error(f"Error configuring Python parameters: {e}")
            return False

    def start_data_fetching_thread(self):
        """Start a background thread to continuously fetch data from the Limelight"""
        def fetch_data_continuously():
            previous_left_line = None
            previous_right_line = None
            error_counter = 0
            max_consecutive_errors = Config.THREADING['max_consecutive_errors']
            error_cooldown = 0
            
            logger.info("Starting data fetching loop")
            while True:
                try:
                    # Get the llpython array directly from NetworkTables
                    llpython = self.nt_client.getNumberArray('llpython', [0] * 8)
                    
                    if len(llpython) >= 8:
                        with self.python_data_lock:
                            # Format from pipeline: 
                            # [left_detected(0/1), x1, y1, x2, y2, right_detected(0/1), x1, y1]
                            
                            # Extract left line data
                            if llpython[0] == 1:  # Left line detected
                                self.latest_left_line = [
                                    int(llpython[1]),  # x1
                                    int(llpython[2]),  # y1
                                    int(llpython[3]),  # x2
                                    int(llpython[4])   # y2
                                ]
                                # Only log when the line changes significantly
                                if previous_left_line is None or not np.array_equal(np.array(self.latest_left_line), np.array(previous_left_line)):
                                    logger.debug(f"Left line detected: {self.latest_left_line}")
                                    previous_left_line = self.latest_left_line
                            else:
                                if previous_left_line is not None:
                                    logger.debug("Left line lost")
                                    previous_left_line = None
                                    
                                self.latest_left_line = None
                            
                            # Extract right line data
                            if llpython[5] == 1:  # Right line detected
                                # For right line, we only get starting point (x1,y1)
                                # Need to calculate end point
                                right_x1 = int(llpython[6])
                                right_y1 = int(llpython[7])
                                
                                # If left line exists, try to make right line parallel to left
                                if self.latest_left_line is not None:
                                    # Get direction vector of left line
                                    left_x1, left_y1, left_x2, left_y2 = self.latest_left_line
                                    left_dx = left_x2 - left_x1
                                    left_dy = left_y2 - left_y1
                                    
                                    # Calculate length and unit vector
                                    left_length = np.sqrt(left_dx**2 + left_dy**2)
                                    if left_length > 0:
                                        # Normalize to unit vector
                                        left_dx /= left_length
                                        left_dy /= left_length
                                        
                                        # Use same length for right line
                                        right_x2 = int(right_x1 + left_dx * left_length)
                                        right_y2 = int(right_y1 + left_dy * left_length)
                                    else:
                                        # If left line has 0 length, extend downward
                                        right_x2 = right_x1
                                        right_y2 = right_y1 + 100
                                else:
                                    # Without left line, extend downward
                                    right_x2 = right_x1
                                    right_y2 = right_y1 + 100
                                
                                self.latest_right_line = [right_x1, right_y1, right_x2, right_y2]
                                
                                # Only log when the line changes significantly
                                if previous_right_line is None or not np.array_equal(np.array(self.latest_right_line), np.array(previous_right_line)):
                                    logger.debug(f"Right line detected: {self.latest_right_line}")
                                    previous_right_line = self.latest_right_line
                            else:
                                if previous_right_line is not None:
                                    logger.debug("Right line lost")
                                    previous_right_line = None
                                    
                                self.latest_right_line = None
                    
                    # Get targeting latency information (only log occasionally)
                    if time.time() % 10 < 0.1:  # Every ~10 seconds
                        pipeline_latency = self.nt_client.getNumber('tl', 0)
                        capture_latency = self.nt_client.getNumber('cl', 0)
                        total_latency = pipeline_latency + capture_latency
                        logger.info(f"Pipeline latency: {pipeline_latency}ms, Capture latency: {capture_latency}ms, Total: {total_latency}ms")
                    
                    # Reset error counter on successful iteration
                    error_counter = 0
                    error_cooldown = 0
                    
                except Exception as e:
                    error_counter += 1
                    
                    # Only log errors if they persist or after cooldown period
                    if error_counter >= max_consecutive_errors or error_cooldown <= 0:
                        logger.error(f"Error fetching NetworkTables data: {e}")
                        error_cooldown = Config.THREADING['error_log_cooldown']
                    
                    error_cooldown -= 1
                
                # Sleep to control update rate
                time.sleep(Config.THREADING['data_polling_rate'])
        
        # Start the thread
        data_thread = threading.Thread(target=fetch_data_continuously, daemon=True)
        data_thread.start()
        logger.info("Started NetworkTables data fetching thread")
        
    def update_detection_params(self, params):
        """Update detection parameters via NetworkTables and HTTP API"""
        try:
            # Update parameters via NetworkTables for camera settings
            if 'exposure' in params:
                self.nt_client.putNumber('exposure', params.get('exposure', Config.CAMERA['exposure']))
            if 'brightness' in params:
                self.nt_client.putNumber('brightness', params.get('brightness', Config.CAMERA['brightness']))
            if 'gain' in params:
                self.nt_client.putNumber('gain', params.get('gain', Config.CAMERA['gain']))
                
            logger.info("Camera parameters updated via NetworkTables")
            
            # For Python-specific parameters, use HTTP
            if 'threshold_min' in params:
                python_params = {
                    'threshold_min': params.get('threshold_min', Config.CAMERA['threshold_min'])
                }
                
                import requests
                response = requests.post(
                    f'http://{self.limelight_address}:5801/pythonconfig',
                    json=python_params,
                    timeout=2
                )
                
                if response.status_code == 200:
                    logger.info("Python parameters updated successfully")
                else:
                    logger.warning(f"Python parameter update failed: {response.status_code}")
                
            return True
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return False
    
    def get_processed_image(self):
        """Get an image with detected lines and green track area"""
        # Get the latest image from Limelight
        raw_image = get_latest_image(self.ll)
        
        if (raw_image is None):
            # Create a fallback image if Limelight image isn't available
            raw_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(raw_image, "No camera feed available", (120, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        # Update the raw image in the web visualizer
        self.web_visualizer.update_raw_image(raw_image)
        
        # Get the latest lines from our data thread
        with self.python_data_lock:
            left_line = self.latest_left_line
            right_line = self.latest_right_line
            
        # Process the frame to highlight detected lines and track area
        processed_image = raw_image.copy()
        
        # Draw track area in green between lines
        if left_line is not None and right_line is not None:
            # Create a polygon from the four points of the two lines
            x1_left, y1_left, x2_left, y2_left = left_line
            x1_right, y1_right, x2_right, y2_right = right_line
            
            # Create a polygon representing the track area
            track_poly = np.array([
                [x1_left, y1_left],
                [x2_left, y2_left],
                [x2_right, y2_right],
                [x1_right, y1_right]
            ], dtype=np.int32)
            
            # Fill the polygon with semi-transparent green
            overlay = processed_image.copy()
            cv2.fillPoly(overlay, [track_poly], (0, 200, 0))  # Green color
            
            # Apply the overlay with transparency
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, processed_image, 1-alpha, 0, processed_image)
        
        # Draw the lines
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(processed_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(processed_image, 'Left Line', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(processed_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(processed_image, 'Right Line', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Add data overlay
        cv2.putText(processed_image, 'Line Detection (Pipeline #1)', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 60
        if left_line is not None:
            cv2.putText(processed_image, f"Left: ({left_line[0]},{left_line[1]}) to ({left_line[2]},{left_line[3]})", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            
        if right_line is not None:
            cv2.putText(processed_image, f"Right: ({right_line[0]},{right_line[1]}) to ({right_line[2]},{right_line[3]})", 
                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            
        # Update the processed image in the web visualizer
        self.web_visualizer.update_processed_image(processed_image, None, None)
        
        return processed_image

    def run(self):
        """Main run loop to process video and update web interface"""
        if not self.setup():
            logger.error("Setup failed. Exiting.")
            return
        
        logger.info("Starting main processing loop")
        try:
            while True:
                # Process the current frame
                processed_image = self.get_processed_image()
                
                # A simplified placeholder for the track map - just a black image with text
                placeholder_map = np.zeros((800, 800, 3), dtype=np.uint8)
                cv2.putText(placeholder_map, "Track mapping disabled", (200, 400), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                self.web_visualizer.update_track_map(placeholder_map)
                
                # Update status message
                with self.python_data_lock:
                    has_left = self.latest_left_line is not None
                    has_right = self.latest_right_line is not None
                
                status = "Detecting lines... "
                if has_left and has_right:
                    status += "Left and right lines detected!"
                elif has_left:
                    status += "Left line detected"
                elif has_right:
                    status += "Right line detected"
                else:
                    status += "No lines detected"
                
                self.web_visualizer.update_mapping_status(
                    distance_to_start=0,
                    distance_traveled=0,
                    status=status,
                    mapping_complete=False
                )
                
                # Sleep to control frame rate
                time.sleep(Config.THREADING['screen_update_interval'])
                
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        finally:
            # Cleanup using the centralized disconnect function
            if self.ll:
                try:
                    disconnect(self.ll, self.nt_client)
                    logger.info("Disconnected from Limelight")
                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")
            logger.info("Program terminated")

if __name__ == "__main__":
    print("\n=================================================")
    print("White Line Detection System (Pipeline #1)")
    print("=================================================")
    
    # Create and run the detector
    detector = LineDetector()
    detector.run()