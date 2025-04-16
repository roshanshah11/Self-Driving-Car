#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Centralized Limelight connection utilities for the autonomous racer
This module provides shared connection, configuration, and utility functions
for all modules that need to connect to the Limelight camera.
"""

import logging
import time
import json
import numpy as np
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("limelight_connection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LimelightConnection")

# Try to import the official Limelight libraries
try:
    import limelight
    import limelightresults
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
            parsed_result = limelightresults.parse_results(result)
            
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

# If this module is run directly, test the connection
if __name__ == "__main__":
    print("\n================================================")
    print("Limelight Connection Utility - Test Mode")
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