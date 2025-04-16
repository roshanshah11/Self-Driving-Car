import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
import time

class AdaptiveRacingLine:
    def __init__(self, num_points=100, car_params=None):
        self.num_points = num_points
        self.car_params = car_params or {
            'max_velocity': 8.0,
            'max_acceleration': 3.0,
            'max_deceleration': 6.0
        }
        # PSO parameters
        self.num_particles = 30
        self.w = 0.5  # Inertia
        self.c1 = 1.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        
        # Empty state initially
        self.centerline = None
        self.racing_line = None
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        
        # For adaptive updates
        self.is_initialized = False
        self.iteration_count = 0
        self.max_iterations = 10  # Per update
        
    def initialize(self, track_map):
        """Initialize PSO particles along the track centerline"""
        # Extract centerline
        self.centerline = self.extract_centerline(track_map)
        self.track_map = track_map
        
        if len(self.centerline) < 10:
            print("Not enough centerline points")
            return False
            
        # Calculate distance transform (for staying on track)
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        self.dist_transform = distance_transform_edt(binary_track)
        
        # Initialize particles
        self.particles = []
        self.particle_best_positions = []
        self.particle_best_fitness = []
        
        # Smooth centerline and use as starting point
        smooth_centerline = self.smooth_path(self.centerline)
        # Resample to desired number of points
        indices = np.linspace(0, len(smooth_centerline)-1, self.num_points).astype(int)
        smooth_centerline = smooth_centerline[indices]
        
        # Initialize particles around the centerline
        for i in range(self.num_particles):
            # Create a particle by perturbing the centerline
            particle = np.copy(smooth_centerline)
            
            # Add random variations - ensure they stay on track
            for j in range(len(particle)):
                point = smooth_centerline[j].astype(int)
                if (0 <= point[0] < track_map.shape[1] and 
                    0 <= point[1] < track_map.shape[0]):
                    # Get local track width from distance transform
                    local_width = self.dist_transform[point[1], point[0]]
                    
                    # Random offset perpendicular to path direction
                    if j < len(smooth_centerline) - 1:
                        tangent = smooth_centerline[(j+1) % len(smooth_centerline)] - smooth_centerline[j]
                    else:
                        tangent = smooth_centerline[j] - smooth_centerline[(j-1) % len(smooth_centerline)]
                    
                    norm = np.linalg.norm(tangent)
                    if norm > 0:
                        tangent = tangent / norm
                        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular vector
                        
                        # Random offset (more conservative to stay on track)
                        offset = np.random.uniform(-0.8, 0.8) * local_width * 0.7
                        particle[j] = smooth_centerline[j] + normal * offset
            
            # Ensure all points are on track
            particle = self.ensure_on_track(particle)
            
            # Initialize velocity as zero
            velocity = np.zeros_like(particle)
            
            # Calculate fitness
            fitness = self.evaluate_racing_line(particle)
            
            self.particles.append({
                'position': particle,
                'velocity': velocity,
                'fitness': fitness
            })
            
            # Initialize particle's best position
            self.particle_best_positions.append(np.copy(particle))
            self.particle_best_fitness.append(fitness)
            
            # Update global best if needed
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = np.copy(particle)
                
        # Set racing line to current best
        self.racing_line = np.copy(self.global_best_position)
        self.is_initialized = True
        return True
        
    def update(self, track_map=None):
        """Run a few PSO iterations to update the racing line"""
        # If not initialized or track map updated, re-initialize
        if track_map is not None:
            # Check if track has changed significantly
            if not self.is_initialized or not np.array_equal(track_map, self.track_map):
                self.initialize(track_map)
        
        if not self.is_initialized:
            return None
            
        start_time = time.time()
        
        # Run a few PSO iterations
        for _ in range(self.max_iterations):
            self.iteration_count += 1
            
            # Update each particle
            for i in range(self.num_particles):
                particle = self.particles[i]
                
                # Update velocity
                r1 = np.random.random(size=(self.num_points, 2))
                r2 = np.random.random(size=(self.num_points, 2))
                
                cognitive = self.c1 * r1 * (self.particle_best_positions[i] - particle['position'])
                social = self.c2 * r2 * (self.global_best_position - particle['position'])
                
                particle['velocity'] = self.w * particle['velocity'] + cognitive + social
                
                # Limit velocity
                max_velocity = 5.0
                velocity_norms = np.linalg.norm(particle['velocity'], axis=1)
                for j in range(len(velocity_norms)):
                    if velocity_norms[j] > max_velocity:
                        particle['velocity'][j] *= (max_velocity / velocity_norms[j])
                
                # Update position
                particle['position'] = particle['position'] + particle['velocity']
                
                # Ensure all points are on track
                particle['position'] = self.ensure_on_track(particle['position'])
                
                # Recalculate fitness
                fitness = self.evaluate_racing_line(particle['position'])
                particle['fitness'] = fitness
                
                # Update particle's best position if improved
                if fitness > self.particle_best_fitness[i]:
                    self.particle_best_fitness[i] = fitness
                    self.particle_best_positions[i] = np.copy(particle['position'])
                    
                    # Update global best if needed
                    if fitness > self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(particle['position'])
            
            # Update racing line
            self.racing_line = np.copy(self.global_best_position)
            
            # Print progress occasionally
            if self.iteration_count % 10 == 0:
                print(f"Iteration {self.iteration_count}: Best fitness = {self.global_best_fitness:.2f}")
                
        processing_time = time.time() - start_time
        # print(f"PSO update took {processing_time:.3f} seconds for {self.max_iterations} iterations")
        
        return self.racing_line
    
    def evaluate_racing_line(self, racing_line):
        """Simplified fitness function for racing line evaluation"""
        if len(racing_line) < 2:
            return float('-inf')
            
        # Factors to consider:
        # 1. Stay on track (most important)
        # 2. Smoothness
        # 3. Taking good racing line (apex)
        
        # Calculate distance from track edge
        edge_distances = np.zeros(len(racing_line))
        off_track_count = 0
        
        for i in range(len(racing_line)):
            x, y = int(racing_line[i, 0]), int(racing_line[i, 1])
            if 0 <= x < self.track_map.shape[1] and 0 <= y < self.track_map.shape[0]:
                if self.track_map[y, x] == 1:
                    edge_distances[i] = self.dist_transform[y, x]
                else:
                    edge_distances[i] = 0.01
                    off_track_count += 1
            else:
                edge_distances[i] = 0.01
                off_track_count += 1
        
        # Heavy penalty for being off track
        off_track_penalty = -1000.0 * off_track_count if off_track_count > 0 else 0.0
        
        # Calculate smoothness (changes in direction)
        smoothness = 0.0
        for i in range(1, len(racing_line)-1):
            v1 = racing_line[i] - racing_line[i-1]
            v2 = racing_line[i+1] - racing_line[i]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0 and v2_norm > 0:
                dot_product = np.dot(v1/v1_norm, v2/v2_norm)
                smoothness += dot_product  # Higher is better
                
        # Calculate curvature to reward proper racing line
        curvature_score = 0.0
        track_center = np.array([self.track_map.shape[1]/2, self.track_map.shape[0]/2])
        
        # Combine factors with weights
        fitness = (
            off_track_penalty +           # Must stay on track
            50.0 * smoothness +           # Reward smoothness
            10.0 * np.sum(edge_distances) # Safety margin from edges
        )
        
        return fitness

    def ensure_on_track(self, points):
        """Make sure all points are on the track"""
        corrected = np.copy(points)
        
        for i in range(len(points)):
            x, y = int(points[i, 0]), int(points[i, 1])
            
            # Check if point is on track
            if (x < 0 or x >= self.track_map.shape[1] or
                y < 0 or y >= self.track_map.shape[0] or
                self.track_map[y, x] != 1):
                
                # Find nearest on-track point
                # Simple spiral search
                found = False
                for radius in range(1, 30):
                    for dx in range(-radius, radius+1):
                        for dy in range(-radius, radius+1):
                            if abs(dx) != radius and abs(dy) != radius:
                                continue
                                
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < self.track_map.shape[1] and
                                0 <= ny < self.track_map.shape[0] and
                                self.track_map[ny, nx] == 1):
                                
                                corrected[i] = np.array([nx, ny], dtype=float)
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                        
                # If no point found, revert to track center (fallback)
                if not found:
                    corrected[i] = np.array([self.track_map.shape[1]/2, 
                                            self.track_map.shape[0]/2])
                    
        return corrected
        
    def extract_centerline(self, track_map):
        """Extract a simple centerline from the track"""
        # Simple distance transform approach
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        
        dist = distance_transform_edt(binary_track)
        ridge = (dist == cv2.dilate(dist, np.ones((5, 5)))) & (dist > 0)
        
        # Get points where the ridge is True
        centerline_points = np.where(ridge)
        centerline = np.column_stack((centerline_points[1], centerline_points[0]))
        
        # If too few points, try a fallback method
        if len(centerline) < 10:
            # Fallback to using contours
            contours, _ = cv2.findContours(binary_track.astype(np.uint8), 
                                         cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_NONE)
            
            if not contours:
                return np.array([[track_map.shape[1]/2, track_map.shape[0]/2]])
                
            largest_contour = max(contours, key=cv2.contourArea)
            contour_points = largest_contour.reshape(-1, 2)
            
            # Calculate center of track
            center_x = np.mean(contour_points[:, 0])
            center_y = np.mean(contour_points[:, 1])
            center = np.array([center_x, center_y])
            
            # Generate centerline by scaling points inward
            centerline = []
            for point in contour_points:
                vec = center - point
                centerline.append(point + 0.5 * vec)
                
            centerline = np.array(centerline)
            
        # Sample points evenly
        if len(centerline) > self.num_points:
            indices = np.linspace(0, len(centerline)-1, self.num_points).astype(int)
            centerline = centerline[indices]
            
        return centerline
    
    def smooth_path(self, points, smoothness=1.0):
        """Apply simple smoothing to a path"""
        if len(points) < 3:
            return points
            
        # Apply a simple moving average
        window_size = min(5, len(points) // 3)
        if window_size < 2:
            return points
            
        smoothed = np.copy(points)
        num_points = len(points)
        
        for i in range(num_points):
            window = []
            for j in range(-window_size//2, window_size//2 + 1):
                idx = (i + j) % num_points  # Wrap around for closed loop
                window.append(points[idx])
                
            smoothed[i] = np.mean(window, axis=0)
            
        return smoothed
        
    def get_racing_line(self):
        """Get the current racing line"""
        if self.racing_line is not None:
            return self.racing_line.astype(np.int32)
        return None
        
    def get_speed_profile(self):
        """Calculate speed profile based on curvature"""
        if self.racing_line is None:
            return None
            
        num_points = len(self.racing_line)
        speeds = np.zeros(num_points)
        max_speed = self.car_params['max_velocity']
        
        # Calculate curvature
        for i in range(num_points):
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            
            v1 = self.racing_line[i] - self.racing_line[prev_idx]
            v2 = self.racing_line[next_idx] - self.racing_line[i]
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                v1 = v1 / v1_norm
                v2 = v2 / v2_norm
                
                # Calculate angle between vectors
                dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angle = np.arccos(dot)
                
                # Higher curvature = slower speed
                segment_length = (v1_norm + v2_norm) / 2
                curvature = angle / max(0.1, segment_length) # Avoid division by zero
                
                if curvature > 0.001:
                    # Simple speed limit: v = k / curvature 
                    # Adjust constant k based on desired aggressiveness
                    k = 2.0 # Lower k = slower in turns
                    speeds[i] = max(1.0, min(max_speed, k / curvature)) 
                else:
                    speeds[i] = max_speed
            else:
                speeds[i] = max_speed
                
        # Smooth speeds
        speeds = self.smooth_array(speeds)
        
        return speeds
    
    def smooth_array(self, array, window=5):
        """Smooth a 1D array using rolling average"""
        result = np.copy(array)
        n = len(array)
        
        for i in range(n):
            window_sum = 0
            count = 0
            for j in range(-window//2, window//2 + 1):
                idx = (i + j) % n
                window_sum += array[idx]
                count += 1
            result[i] = window_sum / count
            
        return result 