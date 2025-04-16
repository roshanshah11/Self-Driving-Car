import os
import json
import time
import numpy as np
import threading
import logging
import cv2
import requests

# Import consolidated line detection modules
from line_detection.line_detector import LineDetector
from line_detection.limelight_core import get_line_data, disconnect
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

class MainApplication:
    def __init__(self):
        # Initialize components
        self.line_detector = LineDetector()
        
        # For storing line detection data
        self.latest_left_line = None
        self.latest_right_line = None
        self.python_data_lock = threading.Lock()
        
        # Initialize web visualizer
        self.web_visualizer = WebVisualizer(
            limelight_manager=None,
            port=Config.NETWORK['web_port']
        )
        
    def setup(self):
        """Setup the system"""
        logger.info("Setting up white line detection system...")
        
        try:
            # Connect and configure the line detector
            if not self.line_detector.connect():
                logger.error("Failed to connect to line detector. Cannot continue.")
                return False
                
            # Update detection parameters with our config values
            self.line_detector.update_detection_params({
                'exposure': Config.CAMERA['exposure'],
                'brightness': Config.CAMERA['brightness'],
                'gain': Config.CAMERA['gain'],
                'threshold_min': Config.CAMERA['threshold_min']
            })
            
            logger.info(f"Successfully connected to Limelight")
            
            # Connect Limelight to web visualizer
            limelight_address = self.line_detector.manager.limelight_address
            self.web_visualizer.set_limelight_address(limelight_address)
            self.web_visualizer.set_limelight_instance(self.line_detector.manager.ll)
            
            # Set up a parameter update callback
            self.web_visualizer.set_param_update_callback(self.update_detection_params)
            
            # Start the data fetching thread
            self.start_data_fetching_thread()
            
            # Start the web visualizer
            self.web_visualizer.start()
            
            logger.info("System setup complete!")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup: {e}")
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
                    # Use our line detector to get the latest lines
                    left_line, right_line = self.line_detector.detect_lines()
                    
                    with self.python_data_lock:
                        self.latest_left_line = left_line
                        self.latest_right_line = right_line
                    
                        # Only log when the lines change significantly
                        if left_line is not None:
                            if previous_left_line is None or not np.array_equal(np.array(left_line), np.array(previous_left_line)):
                                logger.debug(f"Left line detected: {left_line}")
                                previous_left_line = left_line
                        elif previous_left_line is not None:
                            logger.debug("Left line lost")
                            previous_left_line = None
                            
                        if right_line is not None:
                            if previous_right_line is None or not np.array_equal(np.array(right_line), np.array(previous_right_line)):
                                logger.debug(f"Right line detected: {right_line}")
                                previous_right_line = right_line
                        elif previous_right_line is not None:
                            logger.debug("Right line lost")
                            previous_right_line = None
                    
                    # Get targeting latency information (only log occasionally)
                    if time.time() % 10 < 0.1:  # Every ~10 seconds
                        ll = self.line_detector.manager.ll
                        nt_client = self.line_detector.manager.nt_client
                        if nt_client is not None:
                            pipeline_latency = nt_client.getNumber('tl', 0)
                            capture_latency = nt_client.getNumber('cl', 0)
                            total_latency = pipeline_latency + capture_latency
                            logger.info(f"Pipeline latency: {pipeline_latency}ms, Capture latency: {capture_latency}ms, Total: {total_latency}ms")
                    
                    # Reset error counter on successful iteration
                    error_counter = 0
                    error_cooldown = 0
                    
                except Exception as e:
                    error_counter += 1
                    
                    # Only log errors if they persist or after cooldown period
                    if error_counter >= max_consecutive_errors or error_cooldown <= 0:
                        logger.error(f"Error fetching line data: {e}")
                        error_cooldown = Config.THREADING['error_log_cooldown']
                    
                    error_cooldown -= 1
                
                # Sleep to control update rate
                time.sleep(Config.THREADING['data_polling_rate'])
        
        # Start the thread
        data_thread = threading.Thread(target=fetch_data_continuously, daemon=True)
        data_thread.start()
        logger.info("Started data fetching thread")
        
    def update_detection_params(self, params):
        """Update detection parameters via the line detector"""
        try:
            # Update parameters via the line detector
            success = self.line_detector.update_detection_params(params)
            
            if success:
                logger.info("Detection parameters updated successfully")
            else:
                logger.warning("Failed to update some detection parameters")
                
            return success
        except Exception as e:
            logger.error(f"Error updating parameters: {e}")
            return False
    
    def get_processed_image(self):
        """Get an image with detected lines and green track area"""
        # Get the latest image from line detector
        raw_image = self.line_detector.get_latest_image()
        
        if raw_image is None:
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
            # Clean disconnect
            if self.line_detector:
                try:
                    self.line_detector.disconnect()
                    logger.info("Line detector disconnected")
                except Exception as e:
                    logger.error(f"Error during disconnect: {e}")
            logger.info("Program terminated")

if __name__ == "__main__":
    print("\n=================================================")
    print("White Line Detection System (Pipeline #1)")
    print("=================================================")
    
    # Create and run the application
    app = MainApplication()
    app.run()