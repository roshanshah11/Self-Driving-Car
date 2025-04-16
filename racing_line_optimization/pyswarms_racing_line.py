import numpy as np
import cv2
import pyswarms as ps
from scipy.interpolate import splprep, splev
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import distance
import matplotlib.pyplot as plt

class PySwarmsRacingLineOptimizer:
    """
    Racing line optimizer using PySwarms for superior optimization results.
    Specialized for oval tracks with proper racing line characteristics.
    """
    def __init__(self, car_params=None):
        # Default car parameters if none provided
        self.car_params = car_params or {
            'max_velocity': 5.0,        # m/s
            'max_acceleration': 2.0,    # m/s^2
            'max_deceleration': 4.0,    # m/s^2
            'min_turn_radius': 0.5,     # meters
            'wheelbase': 0.3,           # meters
            'mass': 3.0,                # kg
        }
        
    def extract_track_boundaries(self, track_map):
        """Extract inner and outer track boundaries from track map"""
        # Convert to binary if not already
        if len(track_map.shape) > 2:
            # If RGB image, extract green channel
            green_channel = track_map[:,:,1]
            binary_track = np.zeros_like(green_channel)
            binary_track[green_channel > 100] = 1
        else:
            binary_track = track_map.copy()
            binary_track[binary_track > 0] = 1
            
        # Find track boundaries using contour detection
        contours, _ = cv2.findContours(binary_track.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_NONE)
                                     
        if len(contours) == 0:
            print("No track contours found!")
            return None, None
            
        # Get the largest contour (should be the outer boundary)
        outer_boundary = max(contours, key=cv2.contourArea)
        outer_boundary = outer_boundary.reshape(-1, 2)  # Reshape to [x,y] points
        
        # Create distance transform for finding inner boundary
        dist_transform = distance_transform_edt(binary_track)
        
        # Create eroded version to find inner boundary
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(binary_track.astype(np.uint8), kernel, iterations=5)
        
        # Find inner contours
        inner_contours, _ = cv2.findContours(eroded, 
                                          cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_NONE)
        
        if len(inner_contours) == 0:
            # If no inner contour found, approximate one
            # Calculate centroid of the track
            M = cv2.moments(binary_track.astype(np.uint8))
            if M["m00"] == 0:
                centroid = np.array([binary_track.shape[1]//2, binary_track.shape[0]//2])
            else:
                centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                
            # Create inner boundary by moving points toward centroid
            inner_boundary = []
            for point in outer_boundary:
                vec_to_center = centroid - point
                dist_to_center = np.linalg.norm(vec_to_center)
                if dist_to_center > 0:
                    # Move 40% of the way toward center
                    inner_point = point + 0.4 * vec_to_center
                    inner_boundary.append(inner_point)
            
            inner_boundary = np.array(inner_boundary)
        else:
            # Use the largest inner contour
            inner_boundary = max(inner_contours, key=cv2.contourArea)
            inner_boundary = inner_boundary.reshape(-1, 2)
            
        return inner_boundary, outer_boundary
    
    def extract_centerline(self, track_map):
        """Extract the centerline of the track"""
        inner_boundary, outer_boundary = self.extract_track_boundaries(track_map)
        
        if inner_boundary is None or outer_boundary is None:
            print("Could not extract track boundaries")
            return self._create_ellipse_centerline(track_map.shape)
            
        # Calculate track center
        track_center = np.mean(outer_boundary, axis=0)
        
        # Sort boundaries by angle around center for oval tracks
        inner_sorted = self._sort_boundary_by_angle(inner_boundary, track_center)
        outer_sorted = self._sort_boundary_by_angle(outer_boundary, track_center)
        
        # Resample boundaries to have the same number of points
        num_points = min(100, min(len(inner_sorted), len(outer_sorted)))
        inner_resampled = self._resample_boundary(inner_sorted, num_points)
        outer_resampled = self._resample_boundary(outer_sorted, num_points)
        
        # Create centerline by averaging the boundaries
        centerline = []
        for i in range(num_points):
            mid_point = (inner_resampled[i] + outer_resampled[i]) / 2
            centerline.append(mid_point)
            
        centerline = np.array(centerline)
        
        # Smooth the centerline
        centerline_smooth = self._smooth_boundary(centerline)
        
        return centerline_smooth
    
    def _sort_boundary_by_angle(self, boundary, center):
        """Sort boundary points by angle around center"""
        # Calculate angles
        angles = np.arctan2(boundary[:, 1] - center[1], 
                          boundary[:, 0] - center[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_boundary = boundary[sorted_indices]
        
        return sorted_boundary
    
    def _resample_boundary(self, boundary, num_points):
        """Resample boundary to have exactly num_points evenly spaced points"""
        if len(boundary) < 3:
            return boundary
            
        # Calculate cumulative distances
        cumulative_dist = np.zeros(len(boundary))
        for i in range(1, len(boundary)):
            cumulative_dist[i] = cumulative_dist[i-1] + np.linalg.norm(boundary[i] - boundary[i-1])
            
        # Sample points at even intervals
        distances = np.linspace(0, cumulative_dist[-1], num_points)
        
        # Interpolate x and y separately
        resampled_x = np.interp(distances, cumulative_dist, boundary[:, 0])
        resampled_y = np.interp(distances, cumulative_dist, boundary[:, 1])
        
        return np.column_stack((resampled_x, resampled_y))
    
    def _smooth_boundary(self, boundary, smoothness=20):
        """Smooth a boundary using spline interpolation"""
        if len(boundary) < 4:
            return boundary
            
        # Check if boundary is closed
        is_closed = np.linalg.norm(boundary[0] - boundary[-1]) < 20
        
        # Create parameterization based on cumulative distance
        t = np.zeros(len(boundary))
        for i in range(1, len(boundary)):
            t[i] = t[i-1] + np.linalg.norm(boundary[i] - boundary[i-1])
            
        # Normalize parameter to [0, 1]
        if t[-1] > 0:
            t = t / t[-1]
            
        try:
            # Create spline representation
            if is_closed:
                tck, _ = splprep([boundary[:, 0], boundary[:, 1]], s=smoothness, per=1)
            else:
                tck, _ = splprep([boundary[:, 0], boundary[:, 1]], s=smoothness)
                
            # Generate points on the spline
            num_output_points = min(len(boundary) * 2, 200)
            u_new = np.linspace(0, 1, num_output_points)
            smooth_boundary = np.array(splev(u_new, tck)).T
            
            return smooth_boundary
        except Exception as e:
            print(f"Spline smoothing failed: {e}")
            return boundary
    
    def _create_ellipse_centerline(self, shape):
        """Create a simple ellipse centerline as fallback"""
        height, width = shape[0], shape[1]
        center_x, center_y = width // 2, height // 2
        rx, ry = width * 0.35, height * 0.35
        
        t = np.linspace(0, 2*np.pi, 100)
        x = center_x + rx * np.cos(t)
        y = center_y + ry * np.sin(t)
        
        return np.column_stack((x, y))
    
    def optimize_racing_line(self, track_map, centerline=None):
        """
        Generate the optimal racing line using PySwarms optimization.
        
        Parameters:
        - track_map: Binary track map (1 = track, 0 = non-track)
        - centerline: Optional pre-computed centerline
        
        Returns:
        - racing_line: Optimized racing line as a numpy array of points
        """
        # If no centerline provided, extract it
        if centerline is None or len(centerline) == 0:
            centerline = self.extract_centerline(track_map)
            
        if len(centerline) < 4:
            print("Not enough centerline points for optimization")
            return []
            
        # Extract inner and outer track boundaries
        inner_boundary, outer_boundary = self.extract_track_boundaries(track_map)
        
        # If boundary extraction failed, create binary distance transform
        if inner_boundary is None or outer_boundary is None:
            # Create binary track map
            if len(track_map.shape) > 2:
                green_channel = track_map[:,:,1]
                binary_track = np.zeros_like(green_channel)
                binary_track[green_channel > 100] = 1
            else:
                binary_track = track_map.copy()
                binary_track[binary_track > 0] = 1
                
            # Create distance transform (distance to nearest non-track pixel)
            dist_transform = distance_transform_edt(binary_track)
        else:
            # Create a simpler representation for optimization
            binary_track = np.zeros((track_map.shape[0], track_map.shape[1]), dtype=np.uint8)
            if len(track_map.shape) > 2:
                green_channel = track_map[:,:,1]
                binary_track[green_channel > 100] = 1
            else:
                binary_track[track_map > 0] = 1
                
            dist_transform = distance_transform_edt(binary_track)
            
        # Prepare track center coordinates for oval-specific optimizations
        track_center = np.array([track_map.shape[1] // 2, track_map.shape[0] // 2])
        
        # Reduce centerline points for optimization (too many points slow down PySwarms)
        if len(centerline) > 50:
            indices = np.linspace(0, len(centerline) - 1, 50).astype(int)
            centerline_reduced = centerline[indices]
        else:
            centerline_reduced = centerline
            
        # PySwarms requires a flat vector for optimization
        # We'll represent the racing line as offsets from the centerline
        # along the normal direction at each point
        num_points = len(centerline_reduced)
        
        # Calculate normal vectors at each centerline point
        normals = np.zeros((num_points, 2))
        for i in range(num_points):
            prev_idx = (i - 1) % num_points
            next_idx = (i + 1) % num_points
            
            # Calculate tangent as vector from prev to next
            tangent = centerline_reduced[next_idx] - centerline_reduced[prev_idx]
            tangent_norm = np.linalg.norm(tangent)
            
            if tangent_norm > 0:
                tangent = tangent / tangent_norm
                # Normal is perpendicular to tangent
                normals[i] = np.array([-tangent[1], tangent[0]])
                
        # Create bounds for the optimization (offset distances from centerline)
        # PySwarms expects bounds as [lower_bounds, upper_bounds]
        max_offsets = np.zeros(num_points)
        
        for i in range(num_points):
            point = centerline_reduced[i].astype(int)
            if 0 <= point[0] < dist_transform.shape[1] and 0 <= point[1] < dist_transform.shape[0]:
                # Use distance transform to get max possible offset (track width)
                max_offsets[i] = dist_transform[point[1], point[0]] * 0.9  # 90% of distance to edge
            else:
                max_offsets[i] = 10  # Default if point is out of bounds
                
        # For ovals, different constraints for straights vs turns
        # We'll identify turns vs straights based on curvature
        curvature = self._calculate_curvature(centerline_reduced)
        is_turn = np.abs(curvature) > 0.005
        
        # Define bounds - Allow wider variation on straights
        lower_bounds = np.zeros(num_points)
        upper_bounds = np.zeros(num_points)
        
        for i in range(num_points):
            if is_turn[i]:
                # For turns: allow more offset to inside, less to outside
                # Determine inside vs outside based on curvature sign
                if curvature[i] > 0:  # Right turn
                    lower_bounds[i] = -max_offsets[i]
                    upper_bounds[i] = max_offsets[i] * 0.2
                else:  # Left turn
                    lower_bounds[i] = -max_offsets[i] * 0.2
                    upper_bounds[i] = max_offsets[i]
            else:
                # For straights: allow more offset to outside
                # For ovals, we want to stay to the outside on straights
                # But allow some flexibility for transitioning
                lower_bounds[i] = -max_offsets[i] * 0.3
                upper_bounds[i] = max_offsets[i] * 0.7
                
        bounds = (lower_bounds, upper_bounds)
        
        # Initialize swarm
        options = {'c1': 1.5, 'c2': 2.0, 'w': 0.5}
        num_particles = 50
        dimensions = num_points
        
        # Create PySwarms optimizer
        optimizer = ps.single.GlobalBestPSO(
            n_particles=num_particles,
            dimensions=dimensions,
            options=options,
            bounds=bounds
        )
        
        # Define the objective function for PySwarms
        # We'll wrap our evaluation function for the expected PySwarms interface
        def objective_function(offsets):
            # offsets has shape (n_particles, dimensions)
            n_particles = offsets.shape[0]
            fitness_values = np.zeros(n_particles)
            
            for i in range(n_particles):
                # Convert offsets to racing line points
                racing_line_points = self._offsets_to_points(centerline_reduced, normals, offsets[i])
                
                # Evaluate this racing line
                fitness_values[i] = -self._evaluate_racing_line(racing_line_points, binary_track, 
                                                              dist_transform, track_center)
                                                              
            return fitness_values
            
        # Run optimization - we negate fitness because PySwarms minimizes
        print("Starting PySwarms optimization...")
        cost, pos = optimizer.optimize(objective_function, iters=100, verbose=True)
        
        # Convert best position to racing line
        best_offsets = pos
        racing_line = self._offsets_to_points(centerline_reduced, normals, best_offsets)
        
        # Apply final smoothing
        racing_line_smooth = self._smooth_boundary(racing_line, smoothness=15)
        
        # Ensure points are on track
        racing_line_final = self._ensure_on_track(racing_line_smooth, binary_track)
        
        print("PySwarms optimization complete!")
        return racing_line_final
    
    def _offsets_to_points(self, centerline, normals, offsets):
        """Convert centerline + offsets to actual racing line points"""
        racing_line = np.zeros_like(centerline)
        for i in range(len(centerline)):
            # Check for NaN in offsets and replace with 0
            if np.isnan(offsets[i]):
                offsets[i] = 0.0
            racing_line[i] = centerline[i] + offsets[i] * normals[i]
        return racing_line
    
    def _evaluate_racing_line(self, racing_line, binary_track, dist_transform, track_center):
        """
        Evaluate racing line quality. Higher score is better.
        
        Evaluates based on:
        1. Lap time (from curvature & speed profile)
        2. Smoothness
        3. Proper racing line principles (out-in-out)
        4. Safety margin from track edges
        """
        if len(racing_line) < 4:
            return float('-inf')
            
        # Check for NaN values in racing line
        if np.isnan(racing_line).any():
            return float('-inf')  # Return worst possible score for invalid racing lines
            
        # 1. Calculate curvature and lap time
        curvature = self._calculate_curvature(racing_line)
        
        # Calculate distance between consecutive points
        distances = np.zeros(len(racing_line))
        for i in range(1, len(racing_line)):
            distances[i] = np.linalg.norm(racing_line[i] - racing_line[i-1])
            
        # Estimate speeds based on curvature
        max_speed = self.car_params['max_velocity']
        max_lat_accel = 2.0  # m/s^2
        
        speeds = np.zeros(len(racing_line))
        for i in range(len(racing_line)):
            if abs(curvature[i]) > 0.001:
                # v = sqrt(a/k) where k is curvature
                speeds[i] = min(max_speed, np.sqrt(max_lat_accel / max(abs(curvature[i]), 0.001)))
            else:
                speeds[i] = max_speed
                
        # Simple lap time calculation
        pixel_to_meter = 0.01  # 1 pixel = 1 cm
        time_segments = distances[1:] * pixel_to_meter / np.maximum(speeds[1:], 0.01)
        lap_time = np.sum(time_segments)
        
        # 2. Calculate smoothness
        curvature_changes = np.diff(curvature, append=curvature[0])
        smoothness = -np.sum(curvature_changes**2)
        
        # 3. Calculate out-in-out racing principle score
        # Identify turns vs straights
        is_turn = np.abs(curvature) > 0.005
        
        # Group into continuous segments
        segments = []
        current_type = is_turn[0]
        segment_start = 0
        
        for i in range(1, len(is_turn)):
            if is_turn[i] != current_type:
                segments.append((segment_start, i-1, current_type))
                segment_start = i
                current_type = is_turn[i]
                
        # Add the last segment
        segments.append((segment_start, len(is_turn)-1, current_type))
        
        # Score each segment based on racing principles
        racing_principle_score = 0.0
        
        for start, end, is_turning in segments:
            segment_points = racing_line[start:end+1]
            
            # Get average point and vector from track center
            avg_point = np.mean(segment_points, axis=0)
            vec_from_center = avg_point - track_center
            dist_from_center = np.linalg.norm(vec_from_center)
            
            if is_turning:
                # For turns: reward being closer to inside (lower distance from center)
                # This simulates cutting the apex
                racing_principle_score -= dist_from_center * 0.01
            else:
                # For straights: reward being farther from center (wider line)
                racing_principle_score += dist_from_center * 0.01
                
        # 4. Safety margin from track edges
        safety_margin = 0.0
        for point in racing_line:
            # Convert to int safely, checking for NaN or out of bounds
            try:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < binary_track.shape[1] and 0 <= y < binary_track.shape[0]:
                    # Get distance to track edge
                    edge_distance = dist_transform[y, x]
                    
                    # Penalize getting too close to edge (inverse relationship)
                    safety_margin -= 1.0 / max(edge_distance, 0.1)
                    
                    # Hard penalty for off-track points
                    if binary_track[y, x] == 0:
                        safety_margin -= 1000.0
                else:
                    # Out of bounds point
                    safety_margin -= 2000.0
            except (ValueError, TypeError):
                # Handle NaN or other conversion errors
                safety_margin -= 2000.0
                    
        # Check if all points are on track
        on_track_count = 0
        for point in racing_line:
            try:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < binary_track.shape[1] and 0 <= y < binary_track.shape[0]:
                    if binary_track[y, x] > 0:
                        on_track_count += 1
            except (ValueError, TypeError):
                # Skip invalid points
                pass
                    
        on_track_ratio = on_track_count / max(1, len(racing_line))  # Avoid division by zero
        
        # Combined score with weights
        w_time = 1.0
        w_smoothness = 0.5
        w_principles = 0.8
        w_safety = 0.3
        w_on_track = 10.0  # High weight to ensure on-track solution
        
        total_score = (-w_time * lap_time + 
                     w_smoothness * smoothness + 
                     w_principles * racing_principle_score + 
                     w_safety * safety_margin +
                     w_on_track * on_track_ratio)
                     
        return total_score
        
    def _calculate_curvature(self, points):
        """Calculate curvature at each point using circle fitting approach"""
        if len(points) < 3:
            return np.zeros(len(points))
            
        n_points = len(points)
        curvature = np.zeros(n_points)
        
        # Window size for curvature calculation
        window_size = min(7, n_points // 10)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
            
        half_window = window_size // 2
        
        for i in range(n_points):
            # Get indices for window, handling wrap-around for closed loop
            indices = [(i - half_window + j) % n_points for j in range(window_size)]
            window_points = points[indices]
            
            if len(window_points) >= 3:
                try:
                    # Fit circle to window points
                    x, y = window_points[:, 0], window_points[:, 1]
                    
                    # Check for NaN values
                    if np.isnan(x).any() or np.isnan(y).any():
                        raise ValueError("NaN values in window points")
                        
                    # Center the points for numerical stability
                    mean_x, mean_y = np.mean(x), np.mean(y)
                    x_centered = x - mean_x
                    y_centered = y - mean_y
                    
                    # Solve for circle parameters using a more stable approach
                    # Add a small regularization term to improve numerical stability
                    A = np.column_stack([x_centered, y_centered, np.ones(len(x))])
                    b = -x_centered**2 - y_centered**2
                    
                    # Add regularization to prevent numerical instability
                    AT_A = A.T @ A
                    reg = 1e-10 * np.eye(AT_A.shape[0])
                    params = np.linalg.solve(AT_A + reg, A.T @ b)
                    
                    # Calculate radius with safety bounds
                    radius_squared = (params[0]/2)**2 + (params[1]/2)**2 - params[2]
                    # Ensure radius_squared is positive
                    if radius_squared <= 0:
                        radius = 1000.0  # Very large radius for near-straight segments
                    else:
                        radius = np.sqrt(radius_squared)
                    
                    # Curvature = 1/radius
                    if radius > 0.001:
                        # Determine sign based on whether we're turning left or right
                        if i > 0 and i < n_points - 1:
                            prev = points[i-1]
                            curr = points[i]
                            next_pt = points[(i+1) % n_points]
                            
                            v1 = curr - prev
                            v2 = next_pt - curr
                            
                            # Sign from cross product
                            cross = v1[0]*v2[1] - v1[1]*v2[0]
                            sign = -1 if cross < 0 else 1
                            
                            curvature[i] = sign / radius
                        else:
                            curvature[i] = 1.0 / radius
                except Exception as e:
                    # Fallback to simple angle-based estimation
                    if i > 0 and i < n_points - 1:
                        prev = points[i-1]
                        curr = points[i]
                        next_pt = points[(i+1) % n_points]
                        
                        v1 = curr - prev
                        v2 = next_pt - curr
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        
                        if v1_norm > 0 and v2_norm > 0:
                            v1 = v1 / v1_norm
                            v2 = v2 / v2_norm
                            dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
                            angle = np.arccos(dot)
                            
                            # Determine sign based on cross product
                            cross = v1[0]*v2[1] - v1[1]*v2[0]
                            if cross < 0:
                                angle = -angle
                                
                            curvature[i] = angle
                        else:
                            # If vectors have zero norm, set curvature to 0
                            curvature[i] = 0.0
                            
        # Smooth the curvature
        sigma = max(1, n_points // 50)
        return gaussian_filter(curvature, sigma=sigma)
    
    def _ensure_on_track(self, racing_line, binary_track):
        """Make sure all racing line points are on the track"""
        adjusted_line = np.copy(racing_line)
        
        for i in range(len(racing_line)):
            # Skip NaN points
            if np.isnan(racing_line[i, 0]) or np.isnan(racing_line[i, 1]):
                # Replace NaN points with a safe default value (track center)
                adjusted_line[i] = np.array([binary_track.shape[1]//2, binary_track.shape[0]//2])
                continue
                
            try:
                x, y = int(racing_line[i, 0]), int(racing_line[i, 1])
                
                # Check if point is off track or out of bounds
                if (x < 0 or x >= binary_track.shape[1] or 
                    y < 0 or y >= binary_track.shape[0] or
                    binary_track[y, x] == 0):
                    
                    # Find nearest on-track point
                    # Check in expanding radius until we find an on-track point
                    found = False
                    radius = 1
                    max_radius = 20
                    
                    while not found and radius < max_radius:
                        for dy in range(-radius, radius+1):
                            for dx in range(-radius, radius+1):
                                nx, ny = x + dx, y + dy
                                
                                if (0 <= nx < binary_track.shape[1] and 
                                    0 <= ny < binary_track.shape[0] and
                                    binary_track[ny, nx] > 0):
                                    
                                    adjusted_line[i] = np.array([nx, ny])
                                    found = True
                                    break
                                    
                            if found:
                                break
                                
                        radius += 1
                        
                    # If no on-track point found, use track center as fallback
                    if not found:
                        adjusted_line[i] = np.array([binary_track.shape[1]//2, binary_track.shape[0]//2])
            except (ValueError, TypeError):
                # Use track center as fallback for any conversion errors
                adjusted_line[i] = np.array([binary_track.shape[1]//2, binary_track.shape[0]//2])
                
        return adjusted_line
    
    def calculate_speed_profile(self, racing_line):
        """Calculate the optimal speed profile for the racing line"""
        if len(racing_line) < 4:
            return np.ones(len(racing_line)) * self.car_params['max_velocity']
            
        # Calculate curvature along the racing line
        curvature = self._calculate_curvature(racing_line)
        
        # Calculate distance between consecutive points
        distances = np.zeros(len(racing_line))
        for i in range(1, len(racing_line)):
            distances[i] = np.linalg.norm(racing_line[i] - racing_line[i-1])
            
        # Car parameters
        max_v = self.car_params['max_velocity']
        max_a = self.car_params['max_acceleration']
        max_d = self.car_params['max_deceleration']
        max_lat_accel = 2.0  # maximum lateral acceleration (m/s^2)
        
        # Pixel to meter conversion (approximate)
        pixel_to_meter = 0.01  # 1 pixel = 1 cm
        
        # Step 1: Calculate speed limits based on curvature
        speeds = np.zeros(len(racing_line))
        for i in range(len(racing_line)):
            if abs(curvature[i]) > 0.001:
                # v = sqrt(a/k) where k is curvature
                speeds[i] = min(max_v, np.sqrt(max_lat_accel / max(abs(curvature[i]), 0.001)))
            else:
                speeds[i] = max_v
                
        # Step 2: Forward pass - limit by acceleration
        # Start with a reasonable initial speed
        speeds[0] = min(max_v, speeds[0])
        for i in range(1, len(speeds)):
            dist = distances[i] * pixel_to_meter
            v_prev = speeds[i-1]
            v_max_accel = np.sqrt(v_prev**2 + 2 * max_a * dist)
            speeds[i] = min(speeds[i], v_max_accel)
            
        # Step 3: Backward pass - limit by deceleration
        for i in range(len(speeds)-2, -1, -1):
            dist = distances[i+1] * pixel_to_meter
            v_next = speeds[i+1]
            v_max_decel = np.sqrt(v_next**2 + 2 * max_d * dist)
            speeds[i] = min(speeds[i], v_max_decel)
            
        # Smooth the speed profile
        sigma = max(1, len(speeds) // 50)
        smoothed_speeds = gaussian_filter(speeds, sigma=sigma*0.5)
        
        return smoothed_speeds