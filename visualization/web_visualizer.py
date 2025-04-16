import threading
import time
import logging
import json
import numpy as np
import cv2
from flask import Flask, render_template, Response, jsonify

# Set up logging for debugging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("web_visualizer")

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

class WebVisualizer:
    def __init__(self, limelight_manager=None, track_detector=None, racing_optimizer=None, racing_controller=None, port=5000):
        """
        Web-based visualization for the autonomous racer
        Optimized for Raspberry Pi deployment
        
        Args:
            limelight_manager: LimelightManager instance (optional)
            track_detector: TrackDetector instance (optional)
            racing_optimizer: RacingLineOptimizer instance (optional)
            racing_controller: RacingController instance (optional)
            port: Port to run the web server on
        """
        self.limelight_manager = limelight_manager
        self.track_detector = track_detector
        self.racing_optimizer = racing_optimizer
        self.racing_controller = racing_controller
        self.port = port
        self.running = False
        
        # Limelight connection info
        self.limelight_address = None
        self.limelight_instance = None
        
        # For tracking raw images when limelight_manager is None
        self.latest_raw_image = None
        self.latest_processed_image = None
        
        # Create Flask app
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        templates_path = os.path.join(project_root, 'templates')
        static_path = os.path.join(project_root, 'static')
        
        self.app = Flask(__name__, 
                         static_folder=static_path,
                         template_folder=templates_path)
        
        # Enable debug mode for detailed error messages
        self.app.config['DEBUG'] = True
        
        # Store latest data for efficient API responses
        self.latest_data = {
            'track_boundaries': None,
            'racing_line': None,
            'control_commands': {'throttle': 0, 'steering': 0},
            'system_metrics': {'fps': 0, 'cpu': 0, 'memory': 0, 'loop_time': 0},
            'last_update': time.time()
        }
        
        # Rate limit for updates to reduce Raspberry Pi CPU load
        self.update_interval = 0.1  # seconds
        self.last_update_time = 0
        
        # Register routes
        self._register_routes()
        
    def _register_routes(self):
        """Register Flask routes"""
        
        # Set custom JSON encoder for the Flask app
        self.app.json.encoder = NumpyEncoder
        
        @self.app.route('/')
        def index():
            """Serve the main visualization page"""
            # Pass values to the template to enable/disable features based on Raspberry Pi capabilities
            try:
                import platform
                is_raspberry_pi = 'arm' in platform.machine().lower()
            except:
                is_raspberry_pi = False
                
            # Check if we're in simulation mode
            simulation_mode = False
            if self.limelight_manager is not None and hasattr(self.limelight_manager, 'simulation_mode'):
                simulation_mode = self.limelight_manager.simulation_mode
                
            # Default limelight address if not set
            limelight_addr = self.limelight_address or "limelight.local"
                
            return render_template('index.html', 
                                  is_raspberry_pi=is_raspberry_pi,
                                  port=self.port,
                                  simulation_mode=simulation_mode,
                                  limelight_address=limelight_addr)
        
        @self.app.route('/api/track_data')
        def track_data():
            """API endpoint for track data in JSON format"""
            # Only update data if enough time has passed since last update
            # This helps reduce CPU load on Raspberry Pi
            current_time = time.time()
            if current_time - self.last_update_time > self.update_interval:
                self._update_latest_data()
                self.last_update_time = current_time
                
            return jsonify(self.latest_data)
        
        @self.app.route('/video_feed')
        def video_feed():
            """Stream video feed from camera or simulation"""
            return Response(self._generate_video_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
                            
        @self.app.route('/video_processed')
        def video_processed():
            """Stream processed video feed with track boundaries"""
            return Response(self._generate_processed_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
                            
        @self.app.route('/track_map')
        def track_map():
            """Stream track map visualization"""
            return Response(self._generate_map_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')
                            
        @self.app.route('/status')
        def status():
            """Return current status information"""
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
            
            # Get status info from latest_data
            mapping_status = self.latest_data.get('mapping_status', {})
            
            # Return as JSON
            return jsonify({
                'elapsed_time': elapsed_time,
                'distance_to_start': mapping_status.get('distance_to_start', 0),
                'distance_traveled': mapping_status.get('distance_traveled', 0),
                'status': mapping_status.get('status', 'Initializing...'),
                'mapping_complete': mapping_status.get('mapping_complete', False)
            })
            
        @self.app.route('/update_params', methods=['POST'])
        def update_params():
            """Update detection parameters"""
            try:
                # Get parameters from JSON request
                import flask
                params = flask.request.json
                
                # Call the parameter update callback if registered
                if hasattr(self, 'param_update_callback') and self.param_update_callback is not None:
                    success = self.param_update_callback(params)
                    if success:
                        return jsonify({'success': True})
                    else:
                        return jsonify({'success': False, 'message': 'Update failed in callback'})
                else:
                    return jsonify({'success': False, 'message': 'No callback registered'})
            except Exception as e:
                return jsonify({'success': False, 'message': str(e)})
    
    def _generate_video_frames(self):
        """Generate video frames for streaming"""
        while self.running:
            try:
                # Get latest image from limelight or stored image
                if self.limelight_manager is not None:
                    image = self.limelight_manager.get_latest_image()
                else:
                    # Use the latest image set directly
                    image = self.latest_processed_image if self.latest_processed_image is not None else self.latest_raw_image
                
                if image is None:
                    # If no image, create a blank one with info message
                    image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(image, "No camera feed available", (120, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                # Lower resolution for Raspberry Pi to reduce bandwidth and processing
                # Check if we're likely on a Raspberry Pi based on available CPU cores
                try:
                    import multiprocessing
                    if multiprocessing.cpu_count() <= 4:  # Raspberry Pi typically has 4 or fewer cores
                        # Reduce resolution for Pi
                        height, width = image.shape[:2]
                        image = cv2.resize(image, (width//2, height//2))
                except:
                    pass
                
                # Compress image with appropriate quality
                # Lower quality (70) for Raspberry Pi to reduce bandwidth
                ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # Yield the frame for streaming
                frame_data = jpeg.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Add a short sleep to reduce CPU usage
                time.sleep(0.03)  # ~30 FPS max
                
            except Exception as e:
                logger.error(f"Error generating video frame: {e}")
                # Don't crash on error, just try again
                time.sleep(0.1)
    
    def _update_latest_data(self):
        """Update latest data for visualization"""
        try:
            # Get track boundaries with a default empty image
            # This is a temporary fix - we should pass a real image when available
            default_image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            inner, outer = None, None
            if self.track_detector is not None:
                inner, outer = self.track_detector.detect_track_boundaries(default_image)
            
            # Convert track boundaries to serializable format
            track_data = None
            if inner is not None and outer is not None:
                track_data = {
                    'inner': self._contour_to_list(inner),
                    'outer': self._contour_to_list(outer)
                }
            
            # Get racing line if available
            racing_line = None
            if (self.racing_optimizer is not None and 
                hasattr(self.racing_optimizer, 'latest_racing_line') and 
                self.racing_optimizer.latest_racing_line is not None):
                racing_line = self._contour_to_list(self.racing_optimizer.latest_racing_line)
            
            # Get control commands
            throttle, steering = 0, 0
            if (self.racing_controller is not None and
                hasattr(self.racing_controller, 'latest_throttle') and 
                hasattr(self.racing_controller, 'latest_steering')):
                throttle = self.racing_controller.latest_throttle
                steering = self.racing_controller.latest_steering
            
            # Get system metrics if available
            metrics = {'fps': 0, 'cpu': 0, 'memory': 0, 'loop_time': 0}
            if self.racing_controller is not None and hasattr(self.racing_controller, 'metrics'):
                metrics = self.racing_controller.metrics
                
            # Update latest data
            self.latest_data = {
                'track_boundaries': track_data,
                'racing_line': racing_line,
                'control_commands': {'throttle': throttle, 'steering': steering},
                'system_metrics': metrics,
                'last_update': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error updating visualization data: {e}")
    
    def _contour_to_list(self, contour):
        """Convert OpenCV contour to serializable list format"""
        if contour is None:
            return None
            
        # Handle different contour formats
        import numpy as np
        if isinstance(contour, np.ndarray):
            if len(contour.shape) == 3:  # Standard OpenCV contour format (N,1,2)
                return contour.reshape(contour.shape[0], 2).tolist()
            elif len(contour.shape) == 2:  # Already in (N,2) format
                return contour.tolist()
        
        # Handle list of points format
        return [[int(p[0]), int(p[1])] for p in contour]
    
    def start(self):
        """Start the web visualization server"""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()  # Initialize start time for elapsed time tracking
        logger.info(f"Starting web visualization server on port {self.port}")
        
        # Run Flask in a separate thread
        self.flask_thread = threading.Thread(target=self._run_flask)
        self.flask_thread.daemon = True
        self.flask_thread.start()
        
        # Print clear message to console about the web interface URL
        print(f"\n=================================================")
        print(f"🌐 Web interface available at: http://localhost:{self.port}")
        print(f"=================================================\n")
    
    def _run_flask(self):
        """Run Flask server with appropriate settings for Raspberry Pi"""
        try:
            # Use gevent or eventlet for better performance on Raspberry Pi if available
            try:
                try:
                    from gevent.pywsgi import WSGIServer
                    http_server = WSGIServer(('0.0.0.0', self.port), self.app)
                    logger.info("Using gevent WSGI server")
                    http_server.serve_forever()
                except ImportError:
                    logger.warning("gevent is not installed. Falling back to the default Flask server.")
                    self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)
            except ImportError:
                try:
                    import eventlet
                    eventlet.monkey_patch()
                    logger.info("Using eventlet for Flask server")
                    self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)
                except ImportError:
                    # Fall back to the default Flask server
                    logger.info("Using default Flask server - consider installing eventlet or gevent for better performance")
                    self.app.run(host='0.0.0.0', port=self.port, threaded=True, debug=False)
        except Exception as e:
            logger.error(f"Error running Flask server: {e}")
            self.running = False
    
    def stop(self):
        """Stop the web visualization server"""
        logger.info("Stopping web visualization server")
        self.running = False
        
        # The Flask development server doesn't have a clean shutdown method
        # It will be terminated when the main thread exits

    def set_limelight_address(self, address):
        """Set the Limelight camera address"""
        self.limelight_address = address
        logger.info(f"Limelight address set to: {address}")
        
    def set_limelight_instance(self, ll):
        """Set the Limelight instance"""
        self.limelight_instance = ll
        logger.info("Limelight instance set successfully")
        
    def set_param_update_callback(self, callback):
        """Set a callback function to be called when parameters are updated via the web interface"""
        self.param_update_callback = callback
        logger.info("Parameter update callback set successfully")
        
    def update_raw_image(self, image):
        """Update the raw camera image"""
        if hasattr(self, 'limelight_manager') and self.limelight_manager is not None:
            # Store the image in the limelight manager for access in _generate_video_frames
            self.limelight_manager.latest_raw_image = image
            logger.debug("Raw image updated")
        else:
            # Store directly in this class
            self.latest_raw_image = image
            logger.debug("Raw image updated locally")
    
    def update_processed_image(self, image, inner_boundary=None, outer_boundary=None):
        """
        Update the processed image with detected track boundaries for visualization
        
        Args:
            image: The raw image
            inner_boundary: Inner track boundary contour
            outer_boundary: Outer track boundary contour
        """
        if image is None:
            return
            
        # Create a copy for visualization
        vis_image = image.copy()
        
        # Draw car/robot position marker (centered in image)
        height, width = vis_image.shape[:2]
        car_center = (width // 2, height // 2)
        
        # Draw car position as a distinct marker
        cv2.circle(vis_image, car_center, 10, (0, 0, 255), -1)  # Red filled circle
        
        # Add crosshair to indicate car position and orientation
        cv2.line(vis_image, (car_center[0] - 15, car_center[1]), (car_center[0] + 15, car_center[1]), (0, 0, 255), 2)
        cv2.line(vis_image, (car_center[0], car_center[1] - 15), (car_center[0], car_center[1] + 15), (0, 0, 255), 2)
        
        # Add car position label
        cv2.putText(vis_image, "CAR", (car_center[0] + 15, car_center[1] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw the detected boundaries
        if inner_boundary is not None:
            cv2.drawContours(vis_image, [inner_boundary], 0, (0, 255, 0), 2)  # Green
            
        if outer_boundary is not None:
            cv2.drawContours(vis_image, [outer_boundary], 0, (255, 0, 0), 2)  # Red
        
        # Add legend for boundary lines
        legend_y = 30
        cv2.putText(vis_image, "Track Boundaries", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += 30
        cv2.putText(vis_image, "- Green: Inner Boundary", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        legend_y += 25
        cv2.putText(vis_image, "- Red: Outer Boundary", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        legend_y += 25
        cv2.putText(vis_image, "- Red Circle: Car Position", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Store the processed image
        self.latest_processed_image = vis_image
    
    def update_track_map(self, track_map):
        """Update the track map visualization"""
        # Convert numpy array to serializable format
        import numpy as np
        
        # Debug information
        if track_map is None:
            logger.error("Received None track map!")
            return
            
        logger.info(f"Received track map of type {type(track_map)}")
        
        if isinstance(track_map, np.ndarray):
            # Check if it's an image (3D array)
            if len(track_map.shape) == 3:
                logger.info(f"Received track map image with shape {track_map.shape}")
                
                # For now, we don't send the full image through JSON
                # Instead, we'll store it locally and reference it
                self.latest_track_map_image = track_map.copy()  # Make a copy to avoid reference issues
                
                # Check for all-black image (potential issue)
                if np.sum(track_map) == 0:
                    logger.warning("Received all-black track map image!")
                else:
                    logger.info(f"Track map has {np.sum(track_map > 0)} non-zero pixels")
                
                # Just store dimensions and a placeholder in the JSON data
                self.latest_data['track_map'] = {
                    'width': track_map.shape[1],
                    'height': track_map.shape[0],
                    'updated': True
                }
            else:
                # For other arrays, convert to list
                logger.info(f"Received track map array with shape {track_map.shape}")
                self.latest_data['track_map'] = track_map.tolist()
        else:
            # If it's already serializable, store directly
            logger.info(f"Received non-array track map: {type(track_map)}")
            self.latest_data['track_map'] = track_map
        
        logger.info("Track map updated successfully")
    
    def update_mapping_status(self, distance_to_start=None, distance_traveled=None, status=None, mapping_complete=False):
        """Update the mapping status information"""
        # Update mapping status in the latest data
        mapping_status = {
            'distance_to_start': distance_to_start,
            'distance_traveled': distance_traveled,
            'status': status,
            'mapping_complete': mapping_complete
        }
        
        self.latest_data['mapping_status'] = mapping_status
        logger.debug(f"Mapping status updated: {status}")

    def _generate_processed_frames(self):
        """Generate processed video frames with track boundaries for streaming"""
        while self.running:
            try:
                # Use the latest processed image if available
                image = self.latest_processed_image
                
                if image is None:
                    # If no image, create a blank one with info message
                    image = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(image, "No processed image available", (120, 240), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                # Encode image to JPEG format
                ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])
                
                # Yield the frame for streaming
                frame_data = jpeg.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Add a short sleep to reduce CPU usage
                time.sleep(0.1)  # 10 FPS is fine for processed images
                
            except Exception as e:
                logger.error(f"Error generating processed video frame: {e}")
                time.sleep(0.1)
                
    def _generate_map_frames(self):
        """Generate track map visualization frames for streaming"""
        frame_count = 0
        while self.running:
            try:
                frame_count += 1
                # Get the track map from stored image if available
                if hasattr(self, 'latest_track_map_image') and self.latest_track_map_image is not None:
                    image = self.latest_track_map_image.copy()  # Make a copy to avoid reference issues
                    if frame_count % 20 == 0:  # Log every 20 frames
                        logger.info(f"Serving track map frame {frame_count} with shape {image.shape}")
                else:
                    # If no map image is available, create a blank placeholder
                    image = np.zeros((800, 800, 3), dtype=np.uint8)
                    
                    # Add a more informative message
                    cv2.putText(image, "Track map not yet available", (240, 400), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(image, "Waiting for track detection...", (240, 440), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Draw a grid to make it more obvious that the map is working
                    grid_color = (30, 30, 30)  # Dark gray
                    for x in range(0, 800, 100):
                        cv2.line(image, (x, 0), (x, 799), grid_color, 1)
                    for y in range(0, 800, 100):
                        cv2.line(image, (0, y), (799, y), grid_color, 1)
                    
                    if frame_count % 20 == 0:  # Log every 20 frames
                        logger.warning(f"Serving placeholder track map frame {frame_count}")
                
                # Create timestamp overlay for debugging
                import time
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(image, f"Frame: {frame_count} Time: {timestamp}", (10, 790), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Encode image to JPEG format
                ret, jpeg = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 80])
                
                # Yield the frame for streaming
                frame_data = jpeg.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Add a longer sleep - map doesn't need to update as frequently
                time.sleep(0.5)  # 2 FPS is fine for the map
                
            except Exception as e:
                logger.error(f"Error generating map frame: {e}")
                time.sleep(0.5)
