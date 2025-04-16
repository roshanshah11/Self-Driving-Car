import limelight
import limelightresults
import json
import time
import numpy as np
import cv2

# -------------------------------------------------
# Python code to send to the Limelight
# -------------------------------------------------

python_code = '''
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

# -------------------------------------------------
# Connection & Setup Functions
# -------------------------------------------------

def connect_to_limelight():
    """
    Discover and connect to the first available Limelight camera.
    Returns the limelight instance and address if successful.
    """
    discovered_limelights = limelight.discover_limelights(debug=True)
    print("Discovered limelights:", discovered_limelights)

    if not discovered_limelights:
        print("No Limelight cameras found")
        return None, None
        
    limelight_address = discovered_limelights[0] 
    ll = limelight.Limelight(limelight_address)
    
    # Print diagnostic information
    print("-----")
    print("Targeting results:", ll.get_results())
    print("-----")
    print("Status:", ll.get_status())
    print("-----")
    print("Temperature:", ll.get_temp())
    print("-----")
    print("Name:", ll.get_name())
    print("-----")
    print("FPS:", ll.get_fps())
    print("-----")
    print("Hardware report:", ll.hw_report())

    # Enable websocket for better performance
    ll.enable_websocket()
   
    return ll, limelight_address

def configure_limelight_pipeline(ll, limelight_address):
    """
    Configure the Limelight pipeline for white line detection.
    """
    if ll is None:
        return False
        
    # Print the current pipeline settings
    try:
        print(ll.get_pipeline_atindex(0))
    except Exception as e:
        print(f"Error with get_pipeline_atindex: {e}")
        print("Trying HTTP API fallback...")
        try:
            import requests
            pipeline_response = requests.get(f"http://{limelight_address}:5807/pipeline-atindex?index=0")
            if pipeline_response.status_code == 200:
                print(pipeline_response.json())
            else:
                print(f"HTTP fallback failed: {pipeline_response.status_code}")
        except Exception as http_e:
            print(f"HTTP fallback error: {http_e}")

    # Update the current pipeline with white line detection settings
    pipeline_update = {
        'area_max': 98.7,
        'area_min': 1.98778,
        'exposure': 3300.0,   # Exposure time in ms
        'brightness': 60,     # Brightness level (0-100)
        'gain': 50,           # Camera gain (0-100)
        'hue_low': 0,         # HSV threshold values for white
        'hue_high': 180,
        'sat_low': 0,
        'sat_high': 50,
        'val_low': 200,
        'val_high': 255
    }
    
    try:
        ll.update_pipeline(json.dumps(pipeline_update), flush=1)
        print("Pipeline updated successfully")
    except Exception as e:
        print(f"Error with update_pipeline: {e}")
        print("Trying HTTP API fallback...")
        try:
            import requests
            update_response = requests.post(
                f"http://{limelight_address}:5807/update-pipeline?flush=1", 
                json=pipeline_update
            )
            if update_response.status_code == 200:
                print("Pipeline updated via HTTP API")
            else:
                print(f"HTTP update pipeline failed: {update_response.status_code}")
        except Exception as http_e:
            print(f"HTTP fallback error: {http_e}")

    # Verify updated pipeline
    print(ll.get_pipeline_atindex(0))
    
    return True

def upload_python_code(ll, limelight_address, python_code):
    """
    Upload Python code to Limelight Pipeline 1.
    """
    # Switch to pipeline 1
    try:
        ll.pipeline_switch(1)
        print("Switched to pipeline 1")
    except Exception as e:
        print(f"Error with pipeline_switch: {e}")
        print("Trying HTTP API fallback...")
        try:
            import requests
            switch_response = requests.post(f"http://{limelight_address}:5807/pipeline-switch?index=1")
            if switch_response.status_code == 200:
                print("Switched to pipeline 1 via HTTP API")
            else:
                print(f"HTTP pipeline switch failed: {switch_response.status_code}")
        except Exception as http_e:
            print(f"HTTP fallback error: {http_e}")
    
    # Save the Python code to pipeline 1
    try:
        print("Saving Python code to pipeline 1...")
        response = ll.set_python_script(1, python_code)
        print(f"Python code save response: {response}")
        
        # Force save the pipeline to make changes permanent
        ll.save_pipeline(1)
        print("Pipeline 1 saved successfully!")
        
        # Ensure we're still using pipeline 1
        ll.pipeline_switch(1)
        print("Switched to pipeline 1 with the Python code loaded")
        return True
    except Exception as e:
        print(f"Error saving Python code: {e}")
        print("Trying HTTP API fallback...")
        try:
            import requests
            # Upload python code using the upload-python endpoint
            python_response = requests.post(
                f"http://{limelight_address}:5807/upload-python?index=1", 
                data=python_code
            )
            if python_response.status_code == 200:
                print("Python code uploaded via HTTP API")
                
                # Force a reload to apply changes
                reload_response = requests.post(f"http://{limelight_address}:5807/reload-pipeline")
                if reload_response.status_code == 200:
                    print("Pipeline reloaded to apply changes")
                else:
                    print(f"Pipeline reload failed: {reload_response.status_code}")
                
                # Make sure we're still on pipeline 1
                switch_response = requests.post(f"http://{limelight_address}:5807/pipeline-switch?index=1")
                if switch_response.status_code == 200:
                    print("Confirmed pipeline 1 is active via HTTP API")
                    return True
                else:
                    print(f"HTTP switch confirmation failed: {switch_response.status_code}")
                    return False
            else:
                print(f"HTTP python upload failed: {python_response.status_code}")
                return False
        except Exception as http_e:
            print(f"HTTP fallback error: {http_e}")
            return False

# -------------------------------------------------
# Main Function
# -------------------------------------------------

def setup_limelight_for_detection():
    """
    Main function to set up Limelight for white line detection.
    Returns the limelight instance and address if successful.
    """
    # Connect to Limelight
    ll, limelight_address = connect_to_limelight()
    if ll is None:
        return None, None
    
    # Configure the pipeline
    if not configure_limelight_pipeline(ll, limelight_address):
        print("Failed to configure Limelight pipeline")
        return ll, limelight_address
    
    # Upload Python code
    if not upload_python_code(ll, limelight_address, python_code):
        print("Failed to upload Python code")
        return ll, limelight_address
    
    print("Limelight setup complete!")
    return ll, limelight_address

# -------------------------------------------------
# Data Retrieval Functions
# -------------------------------------------------

def get_line_data_from_limelight(ll):
    """
    Get the latest line detection data from Limelight.
    Returns left_line and right_line.
    """
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
        print(f"Error getting line data: {e}")
        return None, None

def get_latest_image(ll):
    """
    Get the latest processed image from Limelight.
    Returns an image with highlighted line detections.
    """
    try:
        # Create base image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get line data
        left_line, right_line = get_line_data_from_limelight(ll)
        
        # If we have line data, draw on the image
        if left_line is not None or right_line is not None:
            # Draw basic visualization instead of using drawDecorations
            if left_line is not None:
                x1, y1, x2, y2 = left_line
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img, 'Left Line', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            
            if right_line is not None:
                x1, y1, x2, y2 = right_line
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, 'Right Line', (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
            return img
        
        # If no lines detected, create generic image
        cv2.putText(img, "No lines detected", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img
        
    except Exception as e:
        print(f"Error getting latest image: {e}")
        # Create error image
        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_img, f"Error: {e}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return error_img

def get_track_boundaries(ll):
    """
    Get track boundaries (inner and outer) from Limelight data.
    Returns inner_boundary, outer_boundary as contours.
    """
    try:
        # Get line data
        left_line, right_line = get_line_data_from_limelight(ll)
        
        # Convert line data to contour format
        inner_boundary = None
        outer_boundary = None
        
        if left_line is not None:
            # Convert left line to inner boundary contour
            x1, y1, x2, y2 = left_line
            inner_boundary = np.array([[[int(x1), int(y1)]], [[int(x2), int(y2)]]], dtype=np.int32)
        
        if right_line is not None:
            # Convert right line to outer boundary contour
            x1, y1, x2, y2 = right_line
            outer_boundary = np.array([[[int(x1), int(y1)]], [[int(x2), int(y2)]]], dtype=np.int32)
        
        return inner_boundary, outer_boundary
    except Exception as e:
        print(f"Error getting track boundaries: {e}")
        return None, None

# Call this function to setup the Limelight when this script is run directly
if __name__ == "__main__":
    ll, limelight_address = setup_limelight_for_detection()
    
    if ll:
        print("Setup successful! Testing data retrieval...")
        try:
            while True:
                # Get and display line data
                left_line, right_line = get_line_data_from_limelight(ll)
                if left_line is not None or right_line is not None:
                    print(f"Left line: {left_line}, Right line: {right_line}")
                else:
                    print("No lines detected")
                    
                # Test image retrieval
                img = get_latest_image(ll)
                if img is not None:
                    print(f"Image shape: {img.shape}")
                    # You could display the image here if needed
                
                time.sleep(1)
        except KeyboardInterrupt:
            print("Program interrupted by user, shutting down.")
        finally:
            try:
                ll.disable_websocket()
                print("Disconnected from Limelight")
            except Exception as e:
                print(f"Error disabling websocket: {e}")
    else:
        print("Setup failed!")
