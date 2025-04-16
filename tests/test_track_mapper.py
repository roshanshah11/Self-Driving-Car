import unittest
import sys
import numpy as np
import cv2
import time
import threading
from main import TrackMapper, Config

class TestTrackMapper(unittest.TestCase):
    def setUp(self):
        # Force simulation mode for all tests
        Config.MAP['size'] = (400, 400)  # Use a smaller map for faster tests
        self.mapper = TrackMapper(use_simulation=True)
        
        # Setup but don't start the web server
        self.original_start = self.mapper.web_visualizer.start
        self.mapper.web_visualizer.start = lambda: None  # No-op to prevent starting the web server
    
    def tearDown(self):
        # Restore original methods
        if hasattr(self, 'original_start'):
            self.mapper.web_visualizer.start = self.original_start
    
    def test_setup(self):
        """Test basic setup functionality"""
        # Run setup
        success = self.mapper.setup()
        
        # Verify setup was successful
        self.assertTrue(success, "Setup should succeed in simulation mode")
        self.assertTrue(self.mapper.use_simulation, "Should be in simulation mode")
        
        # Check that track detector was initialized
        self.assertIsNotNone(self.mapper.track_detector)
        
        # Check that simulated data was created
        with self.mapper.python_data_lock:
            self.assertIsNotNone(self.mapper.latest_left_line)
            self.assertIsNotNone(self.mapper.latest_right_line)
    
    def test_update_pose(self):
        """Test that pose updates work correctly"""
        # Get initial pose
        initial_pose = self.mapper.robot_pose
        
        # Update pose
        self.mapper.update_pose()
        
        # Verify pose changed
        current_pose = self.mapper.robot_pose
        self.assertNotEqual(initial_pose, current_pose, "Robot pose should change after update")
    
    def test_detect_track(self):
        """Test that track detection works"""
        # Setup
        success = self.mapper.setup()
        self.assertTrue(success)
        
        # Detect track
        detection_result = self.mapper.detect_track()
        
        # Verify detection was successful
        self.assertTrue(detection_result, "Track detection should succeed with simulated data")
        
        # Check that track map has some content
        track_map = self.mapper.track_detector.track_map
        self.assertTrue(np.any(track_map > 0), "Track map should contain some track cells")
    
    def test_simulated_track_data(self):
        """Test that simulated track data is generated correctly"""
        # Setup
        self.mapper.robot_pose = (200, 200, 0)  # Set a specific pose
        
        # Create simulated data
        self.mapper.create_simulated_track_data()
        
        # Verify data was created
        with self.mapper.python_data_lock:
            self.assertIsNotNone(self.mapper.latest_left_line)
            self.assertIsNotNone(self.mapper.latest_right_line)
            
            # Check format: [x1, y1, x2, y2]
            self.assertEqual(len(self.mapper.latest_left_line), 4)
            self.assertEqual(len(self.mapper.latest_right_line), 4)
            
            # Left line should be to the left of the robot's position
            # For theta=0, left is -y direction, so y coordinates should be < 200
            self.assertTrue(self.mapper.latest_left_line[1] < 200 or self.mapper.latest_left_line[3] < 200)
            
            # Right line should be to the right of the robot's position
            # For theta=0, right is +y direction, so y coordinates should be > 200
            self.assertTrue(self.mapper.latest_right_line[1] > 200 or self.mapper.latest_right_line[3] > 200)
    
    def test_mapping_integration(self):
        """Test a complete mapping cycle"""
        # Setup
        success = self.mapper.setup()
        self.assertTrue(success)
        
        # Run a few update iterations
        for _ in range(10):
            self.mapper.update_pose()
            self.mapper.detect_track()
            time.sleep(0.1)
        
        # Check that the track map has been populated
        track_map = self.mapper.track_detector.track_map
        self.assertTrue(np.sum(track_map > 0) > 100, "Track map should have significant content after multiple updates")

class TestMapperWithMocks(unittest.TestCase):
    """Tests using mock objects for external dependencies"""
    
    def setUp(self):
        # Create mapper with simulation
        self.mapper = TrackMapper(use_simulation=True)
        
        # Create mock for web visualizer to prevent web server startup
        class MockVisualizer:
            def __init__(self):
                self.raw_image = None
                self.processed_image = None
                self.track_map = None
                self.param_callback = None
            
            def start(self):
                pass
                
            def set_limelight_address(self, address):
                self.limelight_address = address
                
            def set_limelight_instance(self, ll):
                self.ll = ll
                
            def set_param_update_callback(self, callback):
                self.param_callback = callback
                
            def update_raw_image(self, image):
                self.raw_image = image
                
            def update_processed_image(self, image, inner, outer):
                self.processed_image = image
                self.inner = inner
                self.outer = outer
                
            def update_track_map(self, track_map):
                self.track_map = track_map
                
            def update_mapping_status(self, **kwargs):
                self.status = kwargs
        
        # Replace web visualizer with mock
        self.mapper.web_visualizer = MockVisualizer()
    
    def test_parameter_updating(self):
        """Test that parameter updates are correctly handled"""
        # Setup
        self.mapper.setup()
        
        # Create test parameters
        test_params = {
            'exposure': 2000.0,
            'brightness': 60,
            'gain': 40,
            'threshold_min': 160
        }
        
        # Update parameters
        result = self.mapper.update_detection_params(test_params)
        
        # Verify parameters were updated
        self.assertTrue(result, "Parameter update should succeed")
        
        # Check that NetworkTables values were updated (via our mock)
        self.assertEqual(self.mapper.nt_table.getNumber('exposure'), 2000.0)
        self.assertEqual(self.mapper.nt_table.getNumber('brightness'), 60)
        self.assertEqual(self.mapper.nt_table.getNumber('gain'), 40)
    
    def test_visualizer_updates(self):
        """Test that the visualizer is correctly updated"""
        # Setup
        self.mapper.setup()
        
        # Run detection
        self.mapper.detect_track()
        
        # Verify visualizer was updated
        self.assertIsNotNone(self.mapper.web_visualizer.raw_image)
        self.assertIsNotNone(self.mapper.web_visualizer.track_map)
        
        # Should have called update_mapping_status
        self.assertIsNotNone(self.mapper.web_visualizer.status)

if __name__ == "__main__":
    unittest.main() 