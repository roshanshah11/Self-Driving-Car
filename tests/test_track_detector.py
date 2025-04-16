import unittest
import numpy as np
import cv2
from track_detector import TrackDetector

class TestTrackDetector(unittest.TestCase):
    def setUp(self):
        # Create a detector instance for each test
        self.detector = TrackDetector(map_size=(400, 400), cell_size=10)
        
        # Create a simple test image with white lines
        self.test_image = np.zeros((480, 640), dtype=np.uint8)
        # Draw white lines representing track boundaries
        cv2.line(self.test_image, (100, 100), (100, 400), 255, 5)  # Left line
        cv2.line(self.test_image, (500, 100), (500, 400), 255, 5)  # Right line
    
    def test_detect_track_boundaries(self):
        """Test that the detector can find track boundaries in a simple image"""
        # Run detection
        inner_boundary, outer_boundary = self.detector.detect_track_boundaries(self.test_image)
        
        # Assert that boundaries were detected
        self.assertIsNotNone(inner_boundary)
        self.assertIsNotNone(outer_boundary)
        
        # At least one of the boundaries should be close to our lines
        boundaries_valid = False
        
        # Check if inner boundary is close to left line (100, x)
        if inner_boundary is not None:
            inner_x_coords = inner_boundary[:, 0, 0]  # All x coordinates of inner boundary
            if np.any(inner_x_coords < 120) and np.any(inner_x_coords > 80):
                boundaries_valid = True
        
        # Or check if outer boundary is close to right line (500, x)
        if outer_boundary is not None and not boundaries_valid:
            outer_x_coords = outer_boundary[:, 0, 0]  # All x coordinates of outer boundary
            if np.any(outer_x_coords < 520) and np.any(outer_x_coords > 480):
                boundaries_valid = True
        
        self.assertTrue(boundaries_valid, "Detected boundaries don't match expected positions")
    
    def test_update_map(self):
        """Test that updating the map with detected boundaries works"""
        # Create some simple boundary points
        inner_boundary = np.array([[[100, 100]], [[100, 200]], [[100, 300]]], dtype=np.int32)
        outer_boundary = np.array([[[500, 100]], [[500, 200]], [[500, 300]]], dtype=np.int32)
        
        # Set a known robot pose
        robot_pose = (200, 200, 0)  # x, y, theta
        
        # Update the map
        self.detector.update_map(inner_boundary, outer_boundary, robot_pose)
        
        # Get the track map
        track_map = self.detector.track_map
        
        # Verify that some cells in the map are marked as track (value = 1)
        # The exact locations depend on the world-to-map conversion, but there should be some track cells
        self.assertTrue(np.any(track_map == 1), "No track cells were marked in the map")
    
    def test_update_from_line_data(self):
        """Test that updating from line data works"""
        # Create line data similar to what we'd get from the camera
        left_line = [100, 100, 100, 300]  # x1, y1, x2, y2
        right_line = [500, 100, 500, 300]  # x1, y1, x2, y2
        
        # Set a known robot pose
        robot_pose = (200, 200, 0)  # x, y, theta
        
        # Update from line data
        inner_boundary, outer_boundary = self.detector.update_from_line_data(
            left_line, right_line, robot_pose
        )
        
        # Verify that boundaries were returned
        self.assertIsNotNone(inner_boundary)
        self.assertIsNotNone(outer_boundary)
        
        # Verify that the map was updated
        track_map = self.detector.track_map
        self.assertTrue(np.any(track_map == 1), "No track cells were marked in the map")
    
    def test_visualize_track(self):
        """Test that track visualization works"""
        # First update the map with some boundaries
        inner_boundary = np.array([[[100, 100]], [[100, 200]], [[100, 300]]], dtype=np.int32)
        outer_boundary = np.array([[[500, 100]], [[500, 200]], [[500, 300]]], dtype=np.int32)
        self.detector.update_map(inner_boundary, outer_boundary, (200, 200, 0))
        
        # Now visualize the track
        visualization = self.detector.visualize_track()
        
        # Verify that the visualization is an image
        self.assertIsNotNone(visualization)
        self.assertEqual(len(visualization.shape), 3)  # Should be RGB image
        self.assertEqual(visualization.shape[0], 400)  # Height
        self.assertEqual(visualization.shape[1], 400)  # Width
        
        # There should be some non-black pixels in the visualization
        self.assertTrue(np.any(visualization > 0), "Visualization is completely black")

if __name__ == "__main__":
    unittest.main() 