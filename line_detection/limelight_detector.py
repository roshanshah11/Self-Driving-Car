#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limelight camera detector for autonomous racer
This script will detect and connect to the Limelight camera
"""

import json
import time
import sys
from line_detection.limelight_connection import connect_to_limelight, switch_pipeline, set_camera_params, disconnect

def detect_limelight():
    """
    Detect and connect to Limelight camera using the centralized connection utility
    Returns the limelight instance, address, and NetworkTables client if successful
    """
    print("Scanning for Limelight cameras...")
    
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
        print("No Limelight cameras found on the network. Please check:")
        print(" - The camera is properly powered and connected")
        print(" - Your computer is on the same network as the Limelight")
        print(" - The Limelight has a valid IP address")
        
    return ll, limelight_address, nt_client

def configure_limelight(ll, nt_client=None):
    """
    Configure Limelight for white line detection using the centralized utility functions
    """
    if ll is None:
        return False
        
    try:
        # Configure pipeline for white line detection
        print("\nUpdating pipeline for white line detection...")
        
        # Set camera parameters using the centralized utility
        set_camera_params(
            ll=ll, 
            nt_client=nt_client,
            exposure=10.0,    # 10ms exposure
            brightness=75,    # Increase brightness
            gain=50           # Adjust gain
        )
        
        # Update HSV thresholds (currently not supported by set_camera_params)
        # Could be added to the centralized utility in the future
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
        print("Pipeline updated successfully!")
        
        # Switch to pipeline 1 (for Python processing) using the centralized utility
        print("\nSwitching to pipeline 1...")
        switch_pipeline(ll, nt_client, 1)
        print("Now using pipeline 1")
        
        return True
    except Exception as e:
        print(f"Error configuring Limelight: {e}")
        return False

def monitor_limelight(ll, nt_client=None, duration=10):
    """
    Monitor Limelight data for a specified duration in seconds
    """
    if ll is None:
        return
        
    print(f"\nMonitoring Limelight data for {duration} seconds...")
    start_time = time.time()
    
    try:
        # Import here to avoid circular imports
        import limelightresults
        
        while time.time() - start_time < duration:
            # Try to get data through NetworkTables first
            if nt_client is not None:
                try:
                    # Get basic status information
                    tv = nt_client.getNumber('tv', 0)  # Valid targets
                    pipeline = nt_client.getNumber('pipeline', 0)  # Current pipeline
                    latency = nt_client.getNumber('tl', 0)  # Targeting latency

                    print(f"Valid targets: {tv}, Pipeline: {pipeline}, Latency: {latency}ms")
                    
                    # Check for llpython data that would contain line information
                    llpython = nt_client.getNumberArray('llpython', [])
                    if len(llpython) >= 8:
                        if llpython[0] == 1:
                            print(f"  Detected left line: ({llpython[1]:.1f},{llpython[2]:.1f}) to ({llpython[3]:.1f},{llpython[4]:.1f})")
                        if llpython[5] == 1:
                            print(f"  Detected right line: ({llpython[6]:.1f},{llpython[7]:.1f})")
                    
                    # Sleep to control update rate
                    time.sleep(0.5)
                    continue
                except Exception as e:
                    print(f"Error with NetworkTables: {e}, falling back to HTTP API")
                    # Fall through to HTTP API
            
            # Fallback to HTTP API
            result = ll.get_latest_results()
            parsed_result = limelightresults.parse_results(result)
            
            if parsed_result is not None:
                print(f"Valid targets: {parsed_result.validity}, Pipeline: {parsed_result.pipeline_id}, Latency: {parsed_result.targeting_latency}ms")
                
                # Check for retroreflective results (white lines)
                if hasattr(parsed_result, 'retroResults') and parsed_result.retroResults:
                    print(f"  Detected {len(parsed_result.retroResults)} retroreflective targets (white lines)")
                    for i, retro in enumerate(parsed_result.retroResults):
                        print(f"  - Target {i+1}: Area={retro.target_area:.2f}, X={retro.target_x_degrees:.2f}°, Y={retro.target_y_degrees:.2f}°")
            
            # Sleep to control update rate
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    finally:
        print("Monitoring complete")

def main():
    """Main function"""
    print("Limelight Camera Detector")
    print("------------------------")
    
    # Detect and connect to Limelight
    ll, address, nt_client = detect_limelight()
    
    if ll is None:
        print("Failed to connect to Limelight. Exiting.")
        sys.exit(1)
    
    # Configure Limelight
    print("\nConfiguring Limelight...")
    if not configure_limelight(ll, nt_client):
        print("Failed to configure Limelight. Exiting.")
        disconnect(ll, nt_client)
        sys.exit(1)
    
    # Monitor Limelight data
    monitor_limelight(ll, nt_client, duration=10)
    
    # Clean shutdown
    print("\nDisconnecting from Limelight...")
    disconnect(ll, nt_client)
    print("Disconnected successfully")

if __name__ == "__main__":
    main()