#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Limelight core functionality for the autonomous racer
This module provides shared connection, configuration, result parsing and utility functions
for all modules that need to connect to the Limelight camera.
"""

import logging
import time
import json
import numpy as np
import functools
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("limelight_connection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LimelightCore")

# Try to import the official Limelight libraries
try:
    import limelight
    try:
        from networktables import NetworkTables
        NT_AVAILABLE = True
    except ImportError:
        logger.warning("NetworkTables not available. Some features will be limited.")
        NT_AVAILABLE = False
    LIMELIGHT_AVAILABLE = True
except ImportError:
    logger.warning("Limelight libraries not available, running in simulation mode")
    LIMELIGHT_AVAILABLE = False
    NT_AVAILABLE = False

# Default configuration
DEFAULT_CONFIG = {
    'exposure': 1500.0,      # Lower exposure for better white line contrast
    'brightness': 60,        # Brightness for better contrast
    'gain': 50,              # Camera gain
    'pipeline': 1,           # Using pipeline 1 for line detection
    'threshold_min': 200,    # Threshold for white detection
    'request_timeout': 2,    # HTTP request timeout in seconds
    'max_retries': 3,        # Number of connection retries
    'retry_delay': 1,        # Delay between retries in seconds
    'enable_websocket': True # Whether to enable websocket
}

#------------------------------------------------------------------------------
# Result Data Classes
#------------------------------------------------------------------------------

class GeneralResult:
    def __init__(self, results):
        self.barcode = results.get("Barcode", [])
        self.classifierResults = [ClassifierResult(item) for item in results.get("Classifier", [])]
        self.detectorResults = [DetectorResult(item) for item in results.get("Detector", [])]
        self.fiducialResults = [FiducialResult(item) for item in results.get("Fiducial", [])]
        self.retroResults = [RetroreflectiveResult(item) for item in results.get("Retro", [])]
        self.botpose = results.get("botpose", [])
        self.botpose_wpiblue = results.get("botpose_wpiblue", [])
        self.botpose_wpired = results.get("botpose_wpired", [])
        self.capture_latency = results.get("cl", 0)
        self.pipeline_id = results.get("pID", 0)
        self.robot_pose_target_space = results.get("t6c_rs", [])
        self.targeting_latency = results.get("tl", 0)
        self.timestamp = results.get("ts", 0)
        self.validity = results.get("v", 0)
        self.parse_latency = 0.0
        # Store the llpython array for line data
        self.llpython = results.get("llpython", [0, 0, 0, 0, 0, 0, 0, 0])


class RetroreflectiveResult:
    def __init__(self, retro_data):
        self.points = retro_data.get("pts", [])
        self.camera_pose_target_space = retro_data.get("t6c_ts", [])
        self.robot_pose_field_space = retro_data.get("t6r_fs", [])
        self.robot_pose_target_space = retro_data.get("t6r_ts", [])
        self.target_pose_camera_space = retro_data.get("t6t_cs", [])
        self.target_pose_robot_space = retro_data.get("t6t_rs", [])
        self.target_area = retro_data.get("ta", 0)
        self.target_x_degrees = retro_data.get("tx", 0)
        self.target_x_pixels = retro_data.get("txp", 0)
        self.target_y_degrees = retro_data.get("ty", 0)
        self.target_y_pixels = retro_data.get("typ", 0)


class FiducialResult:
    def __init__(self, fiducial_data):
        self.fiducial_id = fiducial_data.get("fID", 0)
        self.family = fiducial_data.get("fam", "")
        self.points = fiducial_data.get("pts", [])
        self.skew = fiducial_data.get("skew", 0)
        self.camera_pose_target_space = fiducial_data.get("t6c_ts", [])
        self.robot_pose_field_space = fiducial_data.get("t6r_fs", [])
        self.robot_pose_target_space = fiducial_data.get("t6r_ts", [])
        self.target_pose_camera_space = fiducial_data.get("t6t_cs", [])
        self.target_pose_robot_space = fiducial_data.get("t6t_rs", [])
        self.target_area = fiducial_data.get("ta", 0)
        self.target_x_degrees = fiducial_data.get("tx", 0)
        self.target_x_pixels = fiducial_data.get("txp", 0)
        self.target_y_degrees = fiducial_data.get("ty", 0)
        self.target_y_pixels = fiducial_data.get("typ", 0)


class DetectorResult:
    def __init__(self, detector_data):
        self.class_name = detector_data.get("class", "")
        self.class_id = detector_data.get("classID", 0)
        self.confidence = detector_data.get("conf", 0)
        self.points = detector_data.get("pts", [])
        self.target_area = detector_data.get("ta", 0)
        self.target_x_degrees = detector_data.get("tx", 0)
        self.target_x_pixels = detector_data.get("txp", 0)
        self.target_y_degrees = detector_data.get("ty", 0)
        self.target_y_pixels = detector_data.get("typ", 0)


class ClassifierResult:
    def __init__(self, classifier_data):
        self.class_name = classifier_data.get("class", "")
        self.class_id = classifier_data.get("classID", 0)
        self.confidence = classifier_data.get("conf", 0)

#------------------------------------------------------------------------------
# Result Parsing Functions
#------------------------------------------------------------------------------

def parse_results(json_data):
    """
    Parse Limelight results JSON data into structured classes
    """
    start_time = time.time()
    if json_data is not None:
        parsed_result = GeneralResult(json_data)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        parsed_result.parse_latency = elapsed_time_ms
        return parsed_result
    return None


def extract_white_lines(parsed_result):
    """
    Extract white line contours from Limelight results
    Specifically designed for track boundary detection
    """
    if parsed_result is None:
        return None, None
    
    # Check for retroreflective results (white lines)
    if parsed_result.retroResults and len(parsed_result.retroResults) >= 2:
        # Sort by area (largest first)
        retro_results = sorted(parsed_result.retroResults, key=lambda x: x.target_area, reverse=True)
        
        # Get the two largest contours (assuming they are inner and outer boundaries)
        outer_boundary = retro_results[0].points if retro_results[0].points else None
        inner_boundary = retro_results[1].points if len(retro_results) > 1 and retro_results[1].points else None
        
        return inner_boundary, outer_boundary
    
    # If no retroreflective results, check detector results as fallback
    elif parsed_result.detectorResults and len(parsed_result.detectorResults) >= 2:
        # Sort by area (largest first)
        detector_results = sorted(parsed_result.detectorResults, key=lambda x: x.target_area, reverse=True)
        
        # Get the two largest contours
        outer_boundary = detector_results[0].points if detector_results[0].points else None
        inner_boundary = detector_results[1].points if len(detector_results) > 1 and detector_results[1].points else None
        
        return inner_boundary, outer_boundary
    
    return None, None

#------------------------------------------------------------------------------
# Limelight Connection Functions
#------------------------------------------------------------------------------

# Custom wrapper for websocket callbacks to prevent parameter mismatch errors
def websocket_callback_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Only pass the first argument to the callback
        # This handles the case when websocket lib passes more args than expected
        return func(args[0])
    return wrapper

def custom_enable_websocket(ll):
    """
    Custom implementation to enable websocket with proper callback handling.
    Wraps the on_close callback to prevent parameter mismatch errors.
    
    Args:
        ll: Limelight instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    if ll is None:
        logger.warning("No Limelight connection available")
        return False
        
    try:
        # Monkey patch the Limelight library's websocket callbacks
        # Store original method to call it later
        original_enable_websocket = ll.enable_websocket
        
        # Access the WebSocket instance and modify its callbacks
        def patched_enable_websocket():
            # Call the original method
            result = original_enable_websocket()
            
            # Get the websocket instance
            if hasattr(ll, '_ws'):
                # Patch the on_close callback with our wrapper
                original_on_close = ll._ws.on_close
                ll._ws.on_close = websocket_callback_wrapper(original_on_close)
                logger.debug("Websocket callbacks patched successfully")
            return result
            
        # Execute the patched method
        patched_enable_websocket()
        logger.info("Websocket enabled with patched callbacks")
        return True
    except Exception as e:
        logger.error(f"Error enabling websocket with patched callbacks: {e}")
        logger.warning("Falling back to HTTP API for data retrieval")
        return False

def custom_disable_websocket(ll):
    """
    Custom implementation to disable websocket with proper error handling.
    
    Args:
        ll: Limelight instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    if ll is None:
        logger.warning("No Limelight connection available")
        return False
        
    try:
        # Check if websocket exists
        if hasattr(ll, '_ws') and ll._ws is not None:
            # Try to close it directly with error handling
            try:
                ll._ws.close()
                logger.info("Websocket closed successfully")
            except Exception as e:
                logger.error(f"Error closing websocket directly: {e}")
                
            # Set to None to make sure it's garbage collected
            ll._ws = None
        
        # Clear any event handlers or callbacks
        if hasattr(ll, '_on_results'):
            ll._on_results = None
            
        return True
    except Exception as e:
        logger.error(f"Error in custom disable websocket: {e}")
        return False

def connect_to_limelight(config=None, use_nt=True, debug=True):
    """
    Connect to the first available Limelight camera.
    
    Args:
        config: Optional configuration dict to override defaults
        use_nt: Whether to use NetworkTables (if available)
        debug: Whether to print debug information
        
    Returns:
        tuple: (limelight_instance, limelight_address, nt_client)
    """
    if not LIMELIGHT_AVAILABLE:
        logger.error("Limelight libraries not available. Please install them with:")
        logger.error("pip install limelight-vision")
        return None, None, None
    
    # Merge provided config with defaults
    if config is None:
        config = DEFAULT_CONFIG
    else:
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        config = merged_config
    
    try:
        # Discover Limelights on the network
        discovered_limelights = limelight.discover_limelights(debug=debug)
        logger.info(f"Discovered limelights: {discovered_limelights}")
        
        if not discovered_limelights:
            logger.error("No Limelight cameras found on the network")
            
            # Try direct HTTP connection if discovery fails
            logger.info("Trying direct HTTP connection...")
            try:
                import requests
                # Try common addresses
                potential_addresses = ["limelight.local", "10.0.0.11", "10.42.0.11", "10.80.3.11"]
                for addr in potential_addresses:
                    try:
                        # Check if we can reach the Limelight status endpoint
                        response = requests.get(f"http://{addr}:5807/status", timeout=config['request_timeout'])
                        if response.status_code == 200:
                            logger.info(f"Found Limelight at {addr} via HTTP API")
                            # Use this address for regular connection
                            ll = limelight.Limelight(addr)
                            limelight_address = addr
                            # Skip discovery since we found one
                            return try_setup_connection(ll, limelight_address, config, use_nt)
                    except Exception:
                        continue
            except ImportError:
                logger.error("requests library not available for HTTP fallback")
            
            logger.error("Failed to discover any Limelight cameras. Cannot continue.")
            logger.error("Please check:")
            logger.error(" - The camera is properly powered and connected")
            logger.error(" - Your computer is on the same network as the Limelight") 
            logger.error(" - The Limelight has a valid IP address")
            logger.error(" - Try running: ping limelight.local")
            return None, None, None
        
        # Connect to the first Limelight found
        limelight_address = discovered_limelights[0]
        ll = limelight.Limelight(limelight_address)
        
        return try_setup_connection(ll, limelight_address, config, use_nt)
    
    except Exception as e:
        logger.error(f"Error connecting to Limelight: {e}")
        return None, None, None

def try_setup_connection(ll, limelight_address, config, use_nt):
    """
    Complete the connection setup after a Limelight is found.
    
    Args:
        ll: Limelight instance
        limelight_address: Limelight address
        config: Configuration dict
        use_nt: Whether to use NetworkTables
        
    Returns:
        tuple: (limelight_instance, limelight_address, nt_client)
    """
    if ll is None:
        return None, None, None
    
    nt_client = None
    
    try:
        # Print diagnostic information
        logger.info(f"Connected to Limelight at {limelight_address}")
        try:
            logger.info(f"Status: {ll.get_status()}")
            logger.info(f"Temperature: {ll.get_temp()}°C")
            logger.info(f"Camera name: {ll.get_name()}")
            logger.info(f"Camera FPS: {ll.get_fps()}")
        except Exception as e:
            logger.warning(f"Could not retrieve all Limelight information: {e}")
        
        # Initialize NetworkTables connection if requested
        if use_nt and NT_AVAILABLE:
            try:
                NetworkTables.initialize(server=limelight_address)
                nt_client = NetworkTables.getTable("limelight")
                logger.info(f"NetworkTables connected to Limelight at {limelight_address}")
            except Exception as e:
                logger.error(f"Failed to initialize NetworkTables: {e}")
                nt_client = None
        
        # Enable websocket for better performance if requested
        if config.get('enable_websocket', True):
            try:
                custom_enable_websocket(ll)
            except Exception as e:
                logger.warning(f"Error enabling websocket: {e}")
                logger.warning("Websocket may not be available, falling back to HTTP API")
        
        # Configure camera settings if provided
        exposure = config.get('exposure')
        brightness = config.get('brightness')
        gain = config.get('gain')
        
        if exposure is not None or brightness is not None or gain is not None:
            try:
                set_camera_params(ll, nt_client, exposure, brightness, gain)
            except Exception as e:
                logger.warning(f"Error setting camera parameters: {e}")
        
        # Switch to the specified pipeline if provided
        pipeline = config.get('pipeline')
        if pipeline is not None:
            try:
                switch_pipeline(ll, nt_client, pipeline)
            except Exception as e:
                logger.warning(f"Error switching pipeline: {e}")
        
        return ll, limelight_address, nt_client
        
    except Exception as e:
        logger.error(f"Error setting up Limelight connection: {e}")
        return ll, limelight_address, None

def set_camera_params(ll, nt_client=None, exposure=None, brightness=None, gain=None):
    """
    Set camera parameters using either NetworkTables or HTTP API.
    
    Args:
        ll: Limelight instance
        nt_client: NetworkTables client (optional)
        exposure: Exposure value in ms
        brightness: Brightness value (0-100)
        gain: Gain value (0-100)
        
    Returns:
        bool: True if successful
    """
    if ll is None:
        logger.warning("No Limelight connection available")
        return False
    
    # Prepare update dict
    pipeline_update = {}
    if exposure is not None:
        pipeline_update['exposure'] = exposure
    if brightness is not None:
        pipeline_update['brightness'] = brightness 
    if gain is not None:
        pipeline_update['gain'] = gain
    
    # Try using NetworkTables first if available
    if nt_client is not None:
        try:
            if exposure is not None:
                nt_client.putNumber('exposure', exposure)
            if brightness is not None:
                nt_client.putNumber('brightness', brightness)
            if gain is not None:
                nt_client.putNumber('gain', gain)
            logger.info(f"Camera settings updated via NetworkTables: {pipeline_update}")
            return True
        except Exception as e:
            logger.warning(f"Error setting camera parameters via NetworkTables: {e}")
            # Fall through to HTTP API
    
    # If no NetworkTables or it failed, use HTTP API
    if pipeline_update:
        try:
            ll.update_pipeline(json.dumps(pipeline_update), flush=1)
            logger.info(f"Camera settings updated via HTTP API: {pipeline_update}")
            return True
        except Exception as e:
            logger.error(f"Error setting camera parameters via HTTP API: {e}")
            try:
                import requests
                limelight_address = ll.hostname if hasattr(ll, 'hostname') else None
                if limelight_address:
                    response = requests.post(
                        f"http://{limelight_address}:5807/update-pipeline?flush=1",
                        json=pipeline_update
                    )
                    if response.status_code == 200:
                        logger.info(f"Camera settings updated via fallback HTTP API: {pipeline_update}")
                        return True
                    else:
                        logger.error(f"HTTP update pipeline failed: {response.status_code}")
            except Exception as http_e:
                logger.error(f"HTTP fallback error: {http_e}")
    
    return False

def switch_pipeline(ll, nt_client=None, pipeline_index=0):
    """
    Switch to a specific pipeline.
    
    Args:
        ll: Limelight instance
        nt_client: NetworkTables client (optional)
        pipeline_index: Pipeline index to switch to
        
    Returns:
        bool: True if successful
    """
    if ll is None:
        logger.warning("No Limelight connection available")
        return False
    
    # Try NetworkTables first if available
    if nt_client is not None:
        try:
            nt_client.putNumber('pipeline', pipeline_index)
            logger.info(f"Switched to pipeline {pipeline_index} via NetworkTables")
            return True
        except Exception as e:
            logger.warning(f"Error switching pipeline via NetworkTables: {e}")
            # Fall through to HTTP API
    
    # If no NetworkTables or it failed, use HTTP API
    try:
        ll.pipeline_switch(pipeline_index)
        logger.info(f"Switched to pipeline {pipeline_index} via HTTP API")
        return True
    except Exception as e:
        logger.error(f"Error switching pipeline via HTTP API: {e}")
        try:
            import requests
            limelight_address = ll.hostname if hasattr(ll, 'hostname') else None
            if limelight_address:
                response = requests.post(
                    f"http://{limelight_address}:5807/pipeline-switch?index={pipeline_index}"
                )
                if response.status_code == 200:
                    logger.info(f"Switched to pipeline {pipeline_index} via fallback HTTP API")
                    return True
                else:
                    logger.error(f"HTTP pipeline switch failed: {response.status_code}")
        except Exception as http_e:
            logger.error(f"HTTP fallback error: {http_e}")
    
    return False

def get_line_data(ll, nt_client=None):
    """
    Get line data from Limelight using either NetworkTables or HTTP API.
    
    Args:
        ll: Limelight instance
        nt_client: NetworkTables client (optional)
        
    Returns:
        tuple: (left_line, right_line)
    """
    # Try NetworkTables first if available (much faster)
    if nt_client is not None:
        try:
            llpython = nt_client.getNumberArray('llpython', [0] * 8)
            
            if len(llpython) >= 8:
                left_line = None
                right_line = None
                
                # Extract left line data
                if llpython[0] == 1:  # Left line detected
                    left_line = [
                        int(llpython[1]),  # x1
                        int(llpython[2]),  # y1
                        int(llpython[3]),  # x2
                        int(llpython[4])   # y2
                    ]
                
                # Extract right line data
                if llpython[5] == 1:  # Right line detected
                    # For right line, we only get starting point (x1,y1)
                    # Need to calculate end point
                    right_x1 = int(llpython[6])
                    right_y1 = int(llpython[7])
                    
                    # If left line exists, try to make right line parallel to left
                    if left_line is not None:
                        # Get direction vector of left line
                        left_x1, left_y1, left_x2, left_y2 = left_line
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
                    
                    right_line = [right_x1, right_y1, right_x2, right_y2]
                
                return left_line, right_line
        
        except Exception as e:
            logger.warning(f"Error getting line data via NetworkTables: {e}")
            # Fall through to HTTP API
    
    # If no NetworkTables or it failed, use HTTP API
    if ll is not None:
        try:
            result = ll.get_latest_results()
            parsed_result = parse_results(result)
            
            if parsed_result is None or not hasattr(parsed_result, 'llpython'):
                return None, None
                
            # Extract line data from llpython array
            llpython = parsed_result.llpython
            
            left_line = None
            right_line = None
            
            # Check if left line is detected (llpython[0] == 1)
            if llpython[0] == 1:
                left_line = llpython[1:5]  # [x1, y1, x2, y2]
                
            # Check if right line is detected (llpython[5] == 1)
            if llpython[5] == 1:
                # We only have the start point (x1, y1) for the right line
                # We need to reconstruct the end point based on line orientation
                x1, y1 = llpython[6], llpython[7]
                
                # If left line exists, make right line parallel
                if left_line is not None:
                    # Get direction vector of left line
                    left_dx = left_line[2] - left_line[0]
                    left_dy = left_line[3] - left_line[1]
                    
                    # Scale to appropriate length
                    length = np.sqrt(left_dx**2 + left_dy**2)
                    if length > 0:
                        left_dx /= length
                        left_dy /= length
                        length = min(length, 200)  # Cap the length
                        
                        # Apply to right line
                        x2 = int(x1 + left_dx * length)
                        y2 = int(y1 + left_dy * length)
                        
                        right_line = [x1, y1, x2, y2]
                    else:
                        # Default fallback if left line has zero length
                        right_line = [x1, y1, x1 + 100, y1]
                else:
                    # Default right line direction if left line doesn't exist
                    right_line = [x1, y1, x1 + 100, y1]
            
            return left_line, right_line
            
        except Exception as e:
            logger.error(f"Error getting line data via HTTP API: {e}")
    
    return None, None

def disconnect(ll, nt_client=None):
    """
    Cleanly disconnect from the Limelight.
    
    Args:
        ll: Limelight instance
        nt_client: NetworkTables client (optional)
        
    Returns:
        bool: True if successful
    """
    success = True
    
    # Disconnect from NetworkTables if connected
    if nt_client is not None and NT_AVAILABLE:
        try:
            NetworkTables.shutdown()
            logger.info("NetworkTables disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from NetworkTables: {e}")
            success = False
    
    # Disconnect from Limelight if connected
    if ll is not None:
        try:
            # Use our custom method to avoid callback parameter issues
            custom_disable_websocket(ll)
            logger.info("Disconnected from Limelight")
        except Exception as e:
            logger.error(f"Error disconnecting from Limelight: {e}")
            success = False
    
    return success

#------------------------------------------------------------------------------
# Limelight Management Class
#------------------------------------------------------------------------------

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
            logger.error("ERROR: Limelight libraries not available. Please install them with:")
            logger.error("pip install limelight-vision")
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
                logger.error("ERROR: Failed to connect to Limelight")
                return False
                
            logger.info(f"Connected to Limelight at {self.limelight_address}")
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Limelight: {e}")
            logger.error("Please verify your Limelight camera is properly connected")
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
            logger.info("Track detection pipeline configured with white line detection optimizations")
            
            # Verify the changes
            updated_pipeline = self.ll.get_pipeline_atindex(0)
            logger.info(f"Updated pipeline settings: {updated_pipeline}")
            
            return True
        except Exception as e:
            logger.error(f"Error configuring pipeline: {e}")
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
            
        logger.info(f"Updating detection parameters: {self.detection_params}")
            
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
                logger.error(f"Error updating Limelight parameters: {e}")
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
            logger.error(f"Failed to get image from Limelight: {e}")
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
            logger.error(f"Failed to get track boundaries: {e}")
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
            logger.error(f"Failed to get IMU data: {e}")
            return None
        
    def disconnect(self):
        """Disconnect from Limelight using the centralized utility"""
        if self.connected and self.ll is not None:
            try:
                # Use the centralized utility
                disconnect(self.ll, self.nt_client)
                logger.info("Disconnected from Limelight")
            except Exception as e:
                logger.error(f"Error disconnecting from Limelight: {e}")
            finally:
                self.connected = False

#------------------------------------------------------------------------------
# Direct Module Functions (for backward compatibility)
#------------------------------------------------------------------------------

def detect_limelight():
    """
    Detect and connect to Limelight camera
    Returns the limelight instance, address, and NetworkTables client if successful
    """
    logger.info("Scanning for Limelight cameras...")
    
    # Specify the configuration
    config = {
        'exposure': 10.0,       # 10ms exposure
        'brightness': 75,       # Increase brightness
        'gain': 50,             # Adjust gain
        'pipeline': 0,          # Default pipeline
        'enable_websocket': True
    }
    
    # Use the centralized connection utility
    ll, limelight_address, nt_client = connect_to_limelight(config=config, use_nt=True, debug=True)
    
    if ll is None:
        logger.warning("No Limelight cameras found on the network. Please check:")
        logger.warning(" - The camera is properly powered and connected")
        logger.warning(" - Your computer is on the same network as the Limelight")
        logger.warning(" - The Limelight has a valid IP address")
        
    return ll, limelight_address, nt_client

def configure_limelight(ll, nt_client=None):
    """
    Configure Limelight for white line detection
    """
    if ll is None:
        return False
        
    try:
        # Configure pipeline for white line detection
        logger.info("\nUpdating pipeline for white line detection...")
        
        # Set camera parameters using the centralized utility
        set_camera_params(
            ll=ll, 
            nt_client=nt_client,
            exposure=10.0,    # 10ms exposure
            brightness=75,    # Increase brightness
            gain=50           # Adjust gain
        )
        
        # Update HSV thresholds
        pipeline_update = {
            # Optimize for white line detection
            'hue_low': 0,
            'hue_high': 180,  # Full hue range
            'sat_low': 0,
            'sat_high': 50,   # Low saturation for white
            'val_low': 180,   # High value for white
            'val_high': 255
        }
        
        # Use direct API call for parameters not yet in centralized utility
        ll.update_pipeline(json.dumps(pipeline_update), flush=1)
        logger.info("Pipeline updated successfully!")
        
        # Switch to pipeline 1 (for Python processing) using the centralized utility
        logger.info("\nSwitching to pipeline 1...")
        switch_pipeline(ll, nt_client, 1)
        logger.info("Now using pipeline 1")
        
        return True
    except Exception as e:
        logger.error(f"Error configuring Limelight: {e}")
        return False

def monitor_limelight(ll, nt_client=None, duration=10):
    """
    Monitor Limelight data for a specified duration in seconds
    """
    if ll is None:
        return
        
    logger.info(f"\nMonitoring Limelight data for {duration} seconds...")
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            # Try to get data through NetworkTables first
            if nt_client is not None:
                try:
                    # Get basic status information
                    tv = nt_client.getNumber('tv', 0)  # Valid targets
                    pipeline = nt_client.getNumber('pipeline', 0)  # Current pipeline
                    latency = nt_client.getNumber('tl', 0)  # Targeting latency

                    logger.info(f"Valid targets: {tv}, Pipeline: {pipeline}, Latency: {latency}ms")
                    
                    # Check for llpython data that would contain line information
                    llpython = nt_client.getNumberArray('llpython', [])
                    if len(llpython) >= 8:
                        if llpython[0] == 1:
                            logger.info(f"  Detected left line: ({llpython[1]:.1f},{llpython[2]:.1f}) to ({llpython[3]:.1f},{llpython[4]:.1f})")
                        if llpython[5] == 1:
                            logger.info(f"  Detected right line: ({llpython[6]:.1f},{llpython[7]:.1f})")
                    
                    # Sleep to control update rate
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    logger.warning(f"Error with NetworkTables: {e}, falling back to HTTP API")
                    # Fall through to HTTP API
            
            # Fallback to HTTP API
            result = ll.get_latest_results()
            parsed_result = parse_results(result)
            
            if parsed_result is not None:
                logger.info(f"Valid targets: {parsed_result.validity}, Pipeline: {parsed_result.pipeline_id}, Latency: {parsed_result.targeting_latency}ms")
                
                # Check for retroreflective results (white lines)
                if hasattr(parsed_result, 'retroResults') and parsed_result.retroResults:
                    logger.info(f"  Detected {len(parsed_result.retroResults)} retroreflective targets (white lines)")
                    for i, retro in enumerate(parsed_result.retroResults):
                        logger.info(f"  - Target {i+1}: Area={retro.target_area:.2f}, X={retro.target_x_degrees:.2f}°, Y={retro.target_y_degrees:.2f}°")
            
            # Sleep to control update rate
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        logger.info("\nMonitoring interrupted by user")
    finally:
        logger.info("Monitoring complete")

# If this module is run directly, test the connection
if __name__ == "__main__":
    print("\n================================================")
    print("Limelight Core Utilities - Test Mode")
    print("================================================\n")
    
    # Test the connection utility
    ll, limelight_address, nt_client = connect_to_limelight(debug=True)
    
    if ll:
        print("\nConnection successful! Testing data retrieval...")
        
        # Test switching pipelines
        switch_pipeline(ll, nt_client, 1)
        
        # Test camera parameter setting
        set_camera_params(ll, nt_client, exposure=1500, brightness=60, gain=50)
        
        try:
            # Test data retrieval in a loop
            for i in range(5):
                left_line, right_line = get_line_data(ll, nt_client)
                if left_line is not None or right_line is not None:
                    print(f"Iteration {i+1}: Left line: {left_line}, Right line: {right_line}")
                else:
                    print(f"Iteration {i+1}: No lines detected")
                time.sleep(1)
                
            print("\nTest complete. Disconnecting...")
            disconnect(ll, nt_client)
            print("Disconnected successfully")
            
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
            disconnect(ll, nt_client)
            print("Disconnected successfully")
            
    else:
        print("Connection failed!")