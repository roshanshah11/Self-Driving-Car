#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line Detection Module
-------------------
Handles white line detection for autonomous racing using Limelight camera.
This consolidated module includes the Limelight detector code and Python code
to be deployed on the Limelight camera for line detection processing.
"""

import cv2
import numpy as np
import time
import sys
# Import consolidated core utilities
from line_detection.limelight_core import (
    LimelightManager,
    get_line_data,
    detect_limelight,
    configure_limelight,
    disconnect
)

#------------------------------------------------------------------------------
# Python code to send to the Limelight camera for line detection
#------------------------------------------------------------------------------

LIMELIGHT_PYTHON_CODE = '''
import cv2
import numpy as np
import time

# Global variables to store the last detected lines
last_left_line = None
last_right_line = None
last_detection_time = 0
line_timeout = 3.0  # How long to keep showing lines after they're no longer detected (seconds)

def detectWhiteLines(image):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Only process the bottom half of the image
    bottom_half = image[height//2:, :]
    
    # Convert to grayscale
    gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 210, 255, cv2.THRESH_BINARY)
    
    # Define regions of interest (ROI) for left and right sides
    bottom_half_height, bottom_half_width = binary.shape
    left_roi = binary[:, :bottom_half_width//2]
    right_roi = binary[:, bottom_half_width//2:]
    
    # Detect edges using Canny
    left_edges = cv2.Canny(left_roi, 50, 150)
    right_edges = cv2.Canny(right_roi, 50, 150)
    
    # Detect lines using Hough Transform
    left_lines = cv2.HoughLinesP(left_edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)
    right_lines = cv2.HoughLinesP(right_edges, 1, np.pi/180, 30, minLineLength=50, maxLineGap=30)
    
    # Get the global variables
    global last_left_line, last_right_line, last_detection_time
    
    left_line = None
    right_line = None
    current_time = time.time()
    
    # Process left lines
    if left_lines is not None and len(left_lines) > 0:
        left_line = left_lines[0][0].copy()  # Make a copy to avoid modifying the original
        # Adjust coordinates to match the full image (add height/2 to y-coordinates)
        left_line[1] += height//2
        left_line[3] += height//2
        last_left_line = left_line.copy()  # Store for persistence
        last_detection_time = current_time
    
    # Process right lines
    if right_lines is not None and len(right_lines) > 0:
        right_line = right_lines[0][0].copy()  # Make a copy to avoid modifying the original
        # Adjust x-coordinates for the right half
        right_line[0] += bottom_half_width//2
        right_line[2] += bottom_half_width//2
        # Adjust y-coordinates for the bottom half
        right_line[1] += height//2
        right_line[3] += height//2
        last_right_line = right_line.copy()  # Store for persistence
        last_detection_time = current_time
    
    # Use the last detected lines if we're still within the timeout period
    if (current_time - last_detection_time) < line_timeout:
        if left_line is None and last_left_line is not None:
            left_line = last_left_line
        if right_line is None and last_right_line is not None:
            right_line = last_right_line
    
    return left_line, right_line

def drawDecorations(image, left_line, right_line):
    visual_image = image.copy()
    height, width = image.shape[:2]
    
    # Add title and processing area indicator
    cv2.putText(visual_image, 'White Line Detection', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(visual_image, (0, height//2), (width, height//2), (255, 255, 255), 1)
    cv2.putText(visual_image, 'Processing Area', (width//2 - 70, height//2 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # If we have both lines, fill the track area between them with GREEN
    if left_line is not None and right_line is not None:
        # Force integers for all coordinates
        lx1, ly1, lx2, ly2 = int(left_line[0]), int(left_line[1]), int(left_line[2]), int(left_line[3])
        rx1, ry1, rx2, ry2 = int(right_line[0]), int(right_line[1]), int(right_line[2]), int(right_line[3])
        
        # Create a polygon for the fill area
        points = np.array([[lx1, ly1], [lx2, ly2], [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        
        # Fill with SOLID green (no transparency)
        mask = np.zeros_like(visual_image)
        cv2.fillPoly(mask, [points], (0, 255, 0))  # Pure green color
        
        # Add the green area to the original image
        visual_image = cv2.addWeighted(visual_image, 1, mask, 0.5, 0)
        
        # Also draw the polygon outline in bright green
        cv2.polylines(visual_image, [points.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
    
    # Draw the left line in BLUE
    if left_line is not None:
        x1, y1, x2, y2 = [int(coord) for coord in left_line]
        cv2.line(visual_image, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Thicker BLUE line
        cv2.putText(visual_image, 'Left Line', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Draw the right line in RED
    if right_line is not None:
        x1, y1, x2, y2 = [int(coord) for coord in right_line]
        cv2.line(visual_image, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Thicker RED line
        cv2.putText(visual_image, 'Right Line', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    
    # Add positional data text
    if left_line is not None or right_line is not None:
        y_pos = 60
        
        if left_line is not None:
            left_text = f"Left: ({left_line[0]:.0f},{left_line[1]:.0f}) to ({left_line[2]:.0f},{left_line[3]:.0f})"
            cv2.putText(visual_image, left_text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 25
            
        if right_line is not None:
            right_text = f"Right: ({right_line[0]:.0f},{right_line[1]:.0f}) to ({right_line[2]:.0f},{right_line[3]:.0f})"
            cv2.putText(visual_image, right_text, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add detection status - shows if we're using cached lines
    global last_detection_time
    current_time = time.time()
    if (current_time - last_detection_time) > 0.2 and (current_time - last_detection_time) < line_timeout:
        status_text = f"Using cached lines ({line_timeout - (current_time - last_detection_time):.1f}s remaining)"
        cv2.putText(visual_image, status_text, (width - 300, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    return visual_image

def runPipeline(image, llrobot):
    largestContour = np.array([[]])
    llpython = [0, 0, 0, 0, 0, 0, 0, 0]
    left_line, right_line = detectWhiteLines(image)
    image = drawDecorations(image, left_line, right_line)
    if left_line is not None:
        llpython[0] = 1
        llpython[1:5] = left_line
    else:
        llpython[0] = 0
    if right_line is not None:
        llpython[5] = 1
        llpython[6] = right_line[0]
        llpython[7] = right_line[1]
    else:
        llpython[5] = 0
    return largestContour, image, llpython
'''

#------------------------------------------------------------------------------
# Line Detector Class 
#------------------------------------------------------------------------------

class LineDetector:
    """
    Line Detector class for detecting white track lines
    This class wraps the Limelight functionality for line detection
    """
    def __init__(self):
        self.manager = LimelightManager()
        self.left_line = None
        self.right_line = None
        self.line_timeout = 3.0
        self.last_detection_time = 0
        self.detected_lines = False
        
        # Default detection parameters
        self.detection_params = {
            'threshold_min': 200,
            'exposure': 1500.0,
            'brightness': 60,
            'gain': 50
        }
        
    def connect(self):
        """Connect to the Limelight camera and initialize line detection"""
        success = self.manager.connect()
        if success:
            # Configure the line detection pipeline
            self.manager.setup_track_detection_pipeline()
            # Update detection parameters
            self.manager.update_detection_params(self.detection_params)
            print("Line detector connected and configured successfully")
        else:
            print("Failed to connect line detector")
        return success
    
    def detect_lines(self):
        """
        Detect white lines using the Limelight camera
        Returns left_line and right_line in format [x1, y1, x2, y2]
        """
        # Get lines from Limelight
        if not self.manager.connected:
            print("Line detector not connected")
            return None, None
            
        left_line, right_line = get_line_data(self.manager.ll, self.manager.nt_client)
        
        # Update detection status
        if left_line is not None or right_line is not None:
            self.detected_lines = True
            self.last_detection_time = time.time()
            self.left_line = left_line
            self.right_line = right_line
        elif self.detected_lines:
            # Check if we're still within the timeout period
            if time.time() - self.last_detection_time < self.line_timeout:
                # Use the last detected lines
                left_line = self.left_line
                right_line = self.right_line
            else:
                # Lines have timed out
                self.detected_lines = False
                
        return left_line, right_line
    
    def get_latest_image(self):
        """Get the latest processed image with lines visualized"""
        return self.manager.get_latest_image()
    
    def update_detection_params(self, params):
        """Update line detection parameters"""
        self.detection_params.update(params)
        return self.manager.update_detection_params(params)
    
    def get_track_boundaries(self):
        """Get track boundaries as contours"""
        return self.manager.get_track_boundaries()
    
    def disconnect(self):
        """Disconnect from the Limelight camera"""
        self.manager.disconnect()
        print("Line detector disconnected")

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

def setup_line_detection(pipeline_index=1):
    """
    Setup the Limelight for line detection
    Returns limelight instance, address, and NetworkTables client
    """
    # Detect and connect to Limelight
    ll, address, nt_client = detect_limelight()
    
    if ll is None:
        print("No Limelight camera found. Cannot setup line detection.")
        return None, None, None
    
    # Configure the Limelight for line detection
    print("Configuring Limelight for line detection...")
    success = configure_limelight(ll, nt_client)
    
    if not success:
        print("Failed to configure Limelight for line detection.")
        disconnect(ll, nt_client)
        return None, None, None
    
    # Upload the Python code to the Limelight
    try:
        print("Uploading Python code to Limelight...")
        ll.write_pipeline_python(pipeline_index, LIMELIGHT_PYTHON_CODE)
        print(f"Python code successfully uploaded to pipeline {pipeline_index}")
        
        # Switch to the pipeline with our Python code
        ll.pipeline_switch(pipeline_index)
        print(f"Switched to pipeline {pipeline_index}")
        
        return ll, address, nt_client
    except Exception as e:
        print(f"Error uploading Python code to Limelight: {e}")
        disconnect(ll, nt_client)
        return None, None, None

#------------------------------------------------------------------------------
# Main function for standalone testing
#------------------------------------------------------------------------------

def main():
    """Main function for testing line detection"""
    print("Line Detector Test")
    print("-----------------")
    
    # Create a line detector
    detector = LineDetector()
    
    # Connect to Limelight
    if not detector.connect():
        print("Failed to connect to Limelight. Exiting.")
        return
    
    try:
        print("Detecting lines (press Ctrl+C to exit)...")
        while True:
            # Detect lines
            left_line, right_line = detector.detect_lines()
            
            if left_line is not None or right_line is not None:
                print("\nLines detected:")
                if left_line is not None:
                    print(f"Left Line: {left_line}")
                if right_line is not None:
                    print(f"Right Line: {right_line}")
            else:
                print(".", end="", flush=True)
                
            # Sleep to control update rate
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Disconnect from Limelight
        detector.disconnect()
        print("Line detector test complete")

if __name__ == "__main__":
    main()