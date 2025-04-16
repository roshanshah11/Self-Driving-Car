import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CameraVisualizer:
    def __init__(self, window_name="Limelight Camera Feed"):
        self.window_name = window_name
        self.raw_window = "Raw Camera Feed"
        self.processed_window = "Processed Image"
        
        # Create windows
        cv2.namedWindow(self.raw_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.processed_window, cv2.WINDOW_NORMAL)
        
        # Resize windows
        cv2.resizeWindow(self.raw_window, 640, 480)
        cv2.resizeWindow(self.processed_window, 640, 480)
        
        # Keep track of images for display
        self.raw_image = None
        self.processed_image = None
        
    def update_raw_image(self, image):
        """Update the raw camera image"""
        if image is not None:
            self.raw_image = image
            # Show image
            cv2.imshow(self.raw_window, self.raw_image)
            cv2.waitKey(1)  # Update display
            
    def update_processed_image(self, image, inner_boundary=None, outer_boundary=None):
        """Update the processed image with boundary overlays"""
        if image is None:
            return
            
        # Make a copy to draw on
        display_image = image.copy()
        
        # Convert to color if grayscale
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
            
        # Draw boundaries if available
        if inner_boundary is not None:
            cv2.drawContours(display_image, [inner_boundary], 0, (0, 255, 0), 2)
        
        if outer_boundary is not None:
            cv2.drawContours(display_image, [outer_boundary], 0, (0, 0, 255), 2)
            
        self.processed_image = display_image
        
        # Show image
        cv2.imshow(self.processed_window, self.processed_image)
        cv2.waitKey(1)  # Update display
        
    def cleanup(self):
        """Close all windows"""
        cv2.destroyAllWindows()