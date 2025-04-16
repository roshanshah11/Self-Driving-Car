import numpy as np
import cv2
import scipy.interpolate as si
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion
from scipy.spatial import distance
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

class RacingLineOptimizer:
    def __init__(self, car_params=None):
        # Default car parameters if none provided
        self.car_params = car_params or {
            'max_velocity': 5.0,  # m/s
            'max_acceleration': 2.0,  # m/s^2
            'max_deceleration': 4.0,  # m/s^2
            'min_turn_radius': 0.5,  # meters
            'wheelbase': 0.3,  # meters
            'mass': 3.0,  # kg
        }
    
    def extract_track_centerline(self, track_map):
        """
        Extract the centerline of the track using distance transform and improved sorting
        for oval track shapes.
        """
        # Get binary version where track = 1, everything else = 0
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        
        # Get distance transform (distance to nearest non-track pixel)
        dist_transform = distance_transform_edt(binary_track)
        
        # Find the ridge of the distance transform (centerline)
        kernel_size = 5
        max_dist = cv2.dilate(dist_transform, np.ones((kernel_size, kernel_size)))
        ridges = (dist_transform == max_dist) & (dist_transform > 0)
        
        # Thin the ridges to get a cleaner centerline
        skeleton = np.zeros_like(track_map)
        skeleton[ridges] = 1
        
        # For ovals, apply additional thinning to get a cleaner centerline
        # This helps with the characteristic shape of oval tracks
        skeleton = binary_erosion(skeleton, structure=np.ones((3, 3)), iterations=1)
        skeleton = binary_erosion(skeleton, structure=np.ones((2, 2)), iterations=1)
        
        # Extract centerline points
        centerline_points = np.where(skeleton > 0)
        centerline_points = np.array(list(zip(centerline_points[1], centerline_points[0])))  # (x, y) format
        
        if len(centerline_points) < 2:
            print("Warning: Not enough centerline points extracted. Using fallback method.")
            return self._extract_centerline_fallback(track_map)
        
        # Find approximate center of the oval
        center_x = np.mean(centerline_points[:, 0])
        center_y = np.mean(centerline_points[:, 1])
        center = np.array([center_x, center_y])
        
        # For oval tracks, angle-based sorting works better than proximity-based
        # Calculate angles from center to each point
        angles = np.zeros(len(centerline_points))
        for i, point in enumerate(centerline_points):
            # Calculate angle from center
            dx = point[0] - center_x
            dy = point[1] - center_y
            angles[i] = np.arctan2(dy, dx)
            
        # Sort points by angle around the center
        sorted_indices = np.argsort(angles)
        sorted_points = centerline_points[sorted_indices]
        
        # Ensure we have enough points for a good centerline
        if len(sorted_points) < 10:
            print("Warning: Not enough sorted centerline points. Using fallback method.")
            return self._extract_centerline_fallback(track_map)
        
        # Create evenly spaced points around the oval
        # This helps with the racing line optimization by providing uniform point distribution
        num_points = min(100, len(sorted_points))
        step = len(sorted_points) // num_points
        
        if step > 0:
            evenly_spaced = sorted_points[::step]
            
            # Ensure we have a closed loop
            if distance.euclidean(evenly_spaced[0], evenly_spaced[-1]) > 20:
                evenly_spaced = np.vstack([evenly_spaced, evenly_spaced[0]])
                
            return evenly_spaced
        else:
            return sorted_points
    
    def _extract_centerline_fallback(self, track_map):
        """
        Fallback method for centerline extraction that works specifically for ovals
        """
        # Get binary version of track
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        
        # Find contours of track
        contours, _ = cv2.findContours(binary_track.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)
        
        # Get the largest contour (the track)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_points = largest_contour.reshape(-1, 2)  # Convert to (x, y) format
            
            # Calculate center of track (average of contour points)
            center_x = np.mean(contour_points[:, 0])
            center_y = np.mean(contour_points[:, 1])
            
            # Generate centerline points by scaling vectors from center
            centerline_points = []
            
            # Calculate centerline for ovals by moving inward from contour
            # For each contour point, move inward toward the center by 50% of the distance
            for point in contour_points:
                vec_to_center = np.array([center_x, center_y]) - point
                # Move 50% of the way toward the center
                centerline_point = point + 0.5 * vec_to_center
                centerline_points.append(centerline_point)
                
            centerline_points = np.array(centerline_points)
            
            # Sort by angle as in the primary method
            angles = np.arctan2(centerline_points[:, 1] - center_y, 
                              centerline_points[:, 0] - center_x)
            sorted_indices = np.argsort(angles)
            sorted_points = centerline_points[sorted_indices]
            
            # Sample points evenly
            num_points = min(100, len(sorted_points))
            indices = np.linspace(0, len(sorted_points)-1, num_points).astype(int)
            evenly_spaced = sorted_points[indices]
            
            return evenly_spaced
        
        # If all else fails, create a simple ellipse as centerline
        else:
            print("Warning: No contours found. Creating a basic oval centerline.")
            
            # Create a basic oval centerline
            height, width = track_map.shape
            center_x = width // 2
            center_y = height // 2
            rx = width * 0.3
            ry = height * 0.3
            
            t = np.linspace(0, 2*np.pi, 100)
            x = center_x + rx * np.cos(t)
            y = center_y + ry * np.sin(t)
            
            return np.column_stack((x, y))
    
    def smooth_path(self, points, smoothness=50):
        """
        Create a smooth path from a set of points using splines
        """
        if len(points) < 4:
            return points
            
        # Convert points to numpy array if not already
        points = np.array(points)
        
        # Check if the path is closed (endpoints are close)
        is_closed = distance.euclidean(points[0], points[-1]) < 30
        
        if (is_closed):
            # For closed paths, we connect the ends
            points = np.vstack([points, points[0]])
        
        # Create a parameterization for the points
        t = np.zeros(len(points))
        for i in range(1, len(points)):
            t[i] = t[i-1] + distance.euclidean(points[i], points[i-1])
            
        # Normalize parameter to [0, 1]
        if t[-1] > 0:
            t = t / t[-1]
        
        # Create the spline representation
        if len(points) > 3:
            # Use a periodic spline for closed paths
            if is_closed:
                tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothness, per=1)
            else:
                tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothness)
                
            # Generate points on the spline
            u_new = np.linspace(0, 1, num=min(500, len(points)*5))
            smooth_points = np.array(splev(u_new, tck)).T
            
            return smooth_points
        else:
            return points
    
    def optimize_racing_line(self, track_map, centerline=None):
        """
        Find the optimal racing line through the track using Particle Swarm Optimization.
        """
        # If no centerline provided, extract it
        if centerline is None or len(centerline) == 0:
            centerline = self.extract_track_centerline(track_map)
            
        if len(centerline) < 4:
            print("Not enough points to optimize racing line")
            return []
        
        # Smooth the centerline to use as a starting point
        smooth_centerline = self.smooth_path(centerline)
        num_points = len(smooth_centerline)
        
        print(f"Starting PSO optimization with {num_points} points...")
        
        # Get track width information for constraints
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        dist_transform = distance_transform_edt(binary_track)
        
        # PSO parameters - IMPROVED FOR OVAL TRACKS
        num_particles = 40  # Increased number of particles for better exploration
        num_iterations = 90  # More iterations for better convergence
        w = 0.55  # Decreased inertia weight for better convergence
        c1 = 1.0  # Cognitive coefficient (personal best) - decreased to favor global best
        c2 = 2.0  # Social coefficient (global best) - increased to favor global best more
        
        # Initialize particles
        particles = []
        particle_best_positions = []
        particle_best_fitness = []
        
        # Global best
        global_best_position = None
        global_best_fitness = float('-inf')
        
        # Initialize particles around the centerline
        for i in range(num_particles):
            # Create a particle by perturbing the centerline
            particle = np.copy(smooth_centerline)
            
            # Add random variation to each point, constrained by track width
            for j in range(len(particle)):
                point = smooth_centerline[j].astype(int)
                if (0 <= point[0] < track_map.shape[1] and 
                    0 <= point[1] < track_map.shape[0]):
                    # Get local track width
                    local_width = dist_transform[point[1], point[0]]
                    
                    # Calculate normal vector (perpendicular to path)
                    if j < len(smooth_centerline) - 1:
                        tangent = smooth_centerline[(j+1) % len(smooth_centerline)] - smooth_centerline[j]
                    else:
                        tangent = smooth_centerline[j] - smooth_centerline[(j-1) % len(smooth_centerline)]
                    
                    norm = np.linalg.norm(tangent)
                    if norm > 0:
                        tangent = tangent / norm
                        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular vector
                        
                        # More conservative perturbation to ensure points stay on track
                        # Use more variation for initial particles for better exploration
                        offset = np.random.uniform(-0.8, 0.8) * local_width * 0.9  # Reduced to 90% of local width
                        particle[j] = smooth_centerline[j] + normal * offset
            
            # Ensure all particle points are on track
            for j in range(len(particle)):
                x, y = int(particle[j, 0]), int(particle[j, 1])
                if (x < 0 or x >= track_map.shape[1] or 
                    y < 0 or y >= track_map.shape[0] or
                    track_map[y, x] != 1):
                    # If off track, find the nearest on-track point
                    # This is better than just reverting to centerline
                    # as it keeps more diversity in the population
                    nearest_on_track = self._find_nearest_on_track_point(particle[j], track_map)
                    particle[j] = nearest_on_track
            
            # Initialize particle velocity as zero
            velocity = np.zeros_like(particle)
            
            # Calculate initial fitness
            fitness = self._evaluate_racing_line(particle, track_map)
            
            particles.append({
                'position': particle,
                'velocity': velocity,
                'fitness': fitness
            })
            
            # Initialize particle's best known position
            particle_best_positions.append(np.copy(particle))
            particle_best_fitness.append(fitness)
            
            # Update global best if needed
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = np.copy(particle)
        
        # Main PSO loop
        for iteration in range(num_iterations):
            for i in range(num_particles):
                particle = particles[i]
                
                # Update velocity with a damping factor to prevent too large movements
                r1 = np.random.random(size=(num_points, 2))
                r2 = np.random.random(size=(num_points, 2))
                
                cognitive_velocity = c1 * r1 * (particle_best_positions[i] - particle['position'])
                social_velocity = c2 * r2 * (global_best_position - particle['position'])
                
                particle['velocity'] = w * particle['velocity'] + cognitive_velocity + social_velocity
                
                # Apply velocity dampening to prevent large jumps out of track
                max_velocity = 5.0  # Set a maximum velocity magnitude
                velocity_norms = np.linalg.norm(particle['velocity'], axis=1)
                for j in range(len(velocity_norms)):
                    if velocity_norms[j] > max_velocity:
                        # Scale down the velocity
                        particle['velocity'][j] = particle['velocity'][j] * (max_velocity / velocity_norms[j])
                
                # Update position
                particle['position'] = particle['position'] + particle['velocity']
                
                # Ensure all particle points are on track - IMPROVED BOUNDARY HANDLING
                for j in range(len(particle['position'])):
                    x, y = int(particle['position'][j, 0]), int(particle['position'][j, 1])
                    
                    # More aggressive boundary checking
                    # Add a small safety margin (1 pixel) to avoid being exactly on the edge
                    if (x < 1 or x >= track_map.shape[1]-1 or 
                        y < 1 or y >= track_map.shape[0]-1 or
                        track_map[y, x] != 1):
                        
                        # Find the nearest on-track point
                        nearest_on_track = self._find_nearest_on_track_point(particle['position'][j], track_map)
                        
                        # Apply the correction with a small random perturbation for diversity
                        particle['position'][j] = nearest_on_track
                        
                        # Reset velocity at this point to avoid bouncing off boundaries
                        particle['velocity'][j] = np.zeros(2)
                
                # Recalculate fitness
                fitness = self._evaluate_racing_line(particle['position'], track_map)
                particle['fitness'] = fitness
                
                # Update particle's best known position if needed
                if fitness > particle_best_fitness[i]:
                    particle_best_fitness[i] = fitness
                    particle_best_positions[i] = np.copy(particle['position'])
                    
                    # Update global best if needed
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = np.copy(particle['position'])
            
            # Print progress
            if iteration % 10 == 0 or iteration == num_iterations - 1:
                print(f"Iteration {iteration+1}/{num_iterations}: Best fitness = {global_best_fitness:.2f}")
            
            # Every 20 iterations, check if any points are off track and fix them
            if iteration % 20 == 0:
                # Pass dist_transform to _ensure_on_track
                global_best_position = self._ensure_on_track(global_best_position, track_map, dist_transform)
        
        # Apply final check to ensure all points are on track
        # Pass dist_transform to _ensure_on_track
        global_best_position = self._ensure_on_track(global_best_position, track_map, dist_transform)
        
        # Apply final smoothing to the best racing line
        try:
            # Use a more conservative smoothing for staying on track
            final_racing_line = self.smooth_path(global_best_position, smoothness=30)
            
            # One more check to ensure smoothed line is on track with margin
            # Pass dist_transform to _ensure_on_track
            final_racing_line = self._ensure_on_track(final_racing_line, track_map, dist_transform)
            
            print("PSO optimization complete!")
            return final_racing_line
        except ValueError:
            # If smoothing fails, try a more robust smoothing approach
            print("Standard smoothing failed, trying robust smoothing...")
            try:
                # Use a more robust smoothing method with error checking
                final_racing_line = self._robust_smooth_path(global_best_position)
                
                # Ensure the robust smoothed line is on track with margin
                # Pass dist_transform to _ensure_on_track
                final_racing_line = self._ensure_on_track(final_racing_line, track_map, dist_transform)
                
                print("PSO optimization complete with robust smoothing!")
                return final_racing_line
            except Exception as e:
                print(f"Error in robust smoothing: {e}")
                # If all smoothing fails, return the unsmoothed best position after ensuring it's on track
                print("Returning unsmoothed racing line")
                # Pass dist_transform to _ensure_on_track
                return self._ensure_on_track(global_best_position, track_map, dist_transform)
    
    def _find_nearest_on_track_point(self, point, track_map):
        """
        Find the nearest point on track to the given point.
        Uses a spiral search pattern to find the closest on-track point.
        """
        x, y = int(point[0]), int(point[1])
        
        # If already on track, return the point
        if (0 <= x < track_map.shape[1] and 
            0 <= y < track_map.shape[0] and 
            track_map[y, x] == 1):
            return point
        
        # Set a maximum search radius
        max_radius = 20  # pixels
        
        # Spiral search for the nearest on-track point
        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    # Skip points not on the radius
                    if abs(dx) != radius and abs(dy) != radius:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    
                    # Check if the point is valid and on track
                    if (0 <= nx < track_map.shape[1] and 
                        0 <= ny < track_map.shape[0] and 
                        track_map[ny, nx] == 1):
                        return np.array([nx, ny], dtype=float)
        
        # If no point found, return the original point clamped to track boundaries
        clamped_x = np.clip(x, 0, track_map.shape[1] - 1)
        clamped_y = np.clip(y, 0, track_map.shape[0] - 1)
        
        # Find the center of the track (approximate)
        center_x = track_map.shape[1] / 2
        center_y = track_map.shape[0] / 2
        
        # Move toward center if still not on track
        vec_to_center = np.array([center_x, center_y]) - np.array([clamped_x, clamped_y])
        norm = np.linalg.norm(vec_to_center)
        if norm > 0:
            vec_to_center = vec_to_center / norm
            
            # Try multiple distances toward center
            for d in range(1, max_radius + 1):
                test_x = int(clamped_x + d * vec_to_center[0])
                test_y = int(clamped_y + d * vec_to_center[1])
                
                if (0 <= test_x < track_map.shape[1] and 
                    0 <= test_y < track_map.shape[0] and 
                    track_map[test_y, test_x] == 1):
                    return np.array([test_x, test_y], dtype=float)
        
        # Fallback to center of track
        return np.array([center_x, center_y], dtype=float)
    
    def _ensure_on_track(self, points, track_map, dist_transform, safety_margin=1.5):
        """
        Ensures all points in the racing line are on the track with a minimum safety margin.
        """
        corrected_points = np.copy(points)
        height, width = track_map.shape
        
        for i in range(len(points)):
            x, y = int(points[i, 0]), int(points[i, 1])
            
            # Check if point is outside bounds or too close to edge
            is_off_track = False
            if (x < 0 or x >= width or 
                y < 0 or y >= height):
                is_off_track = True
            else:
                # Check track map first (faster)
                if track_map[y, x] != 1:
                    is_off_track = True
                # Then check distance transform for safety margin
                elif dist_transform[y, x] <= safety_margin:
                    is_off_track = True
            
            if is_off_track:
                # Find nearest on-track point (ensure it respects margin if possible)
                # Note: _find_nearest_on_track_point might not guarantee the margin, 
                # but it's better than leaving the point off-track.
                corrected_points[i] = self._find_nearest_on_track_point(points[i], track_map)
        
        return corrected_points

    def adjust_for_curvature(self, centerline, track_map):
        """
        Adjust the racing line based on curvature of the centerline.
        Shifts points toward the inside of curves proportional to curvature.
        Modified specifically for oval tracks.
        """
        if len(centerline) < 4:
            return centerline

        # Calculate local curvature at each point using a 3-point window
        curvature = np.zeros(len(centerline))
        
        # Use a larger window to calculate curvature more accurately for ovals
        window_size = 7  # Increased from 5 for smoother curvature detection
        half_window = window_size // 2

        for i in range(len(centerline)):
            # Get points before and after current point (wrap around for closed loop)
            indices = [(i - half_window + j) % len(centerline) for j in range(window_size)]
            window_points = centerline[indices]
            
            # Fit a circle to the window points if we have enough points
            if len(window_points) >= 3:
                try:
                    # Estimate curvature from circle fit (advanced approach)
                    x = window_points[:, 0]
                    y = window_points[:, 1]
                    
                    # Center the points (improves numerical stability)
                    mean_x = np.mean(x)
                    mean_y = np.mean(y)
                    x_centered = x - mean_x
                    y_centered = y - mean_y
                    
                    # Solve for circle parameters
                    A = np.column_stack([x_centered, y_centered, np.ones(len(x))])
                    b = -x_centered**2 - y_centered**2
                    
                    # Use least squares to solve for circle parameters
                    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Calculate circle center and radius
                    center_x = -params[0]/2 + mean_x
                    center_y = -params[1]/2 + mean_y
                    radius = np.sqrt((params[0]/2)**2 + (params[1]/2)**2 - params[2])
                    
                    # Curvature = 1/radius
                    if radius > 0.001:  # Avoid division by zero
                        # Determine sign of curvature based on cross product
                        if i+1 < len(centerline) and i > 0:
                            v1 = centerline[i] - centerline[i-1]
                            v2 = centerline[i+1] - centerline[i]
                            cross = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                            sign = -1 if cross < 0 else 1
                            curvature[i] = sign / radius
                        else:
                            curvature[i] = 1 / radius
                except:
                    # Fallback to simple angle-based curvature if circle fitting fails
                    if i > 0 and i < len(centerline) - 1:
                        prev = centerline[i-1]
                        curr = centerline[i]
                        next_pt = centerline[i+1]
                        v1 = curr - prev
                        v2 = next_pt - curr
                        v1_norm = np.linalg.norm(v1)
                        v2_norm = np.linalg.norm(v2)
                        if v1_norm > 0 and v2_norm > 0:
                            v1_unit = v1 / v1_norm
                            v2_unit = v2 / v2_norm
                            dot_product = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
                            angle = np.arccos(dot_product)
                            # Determine curve direction
                            cross = np.cross(np.append(v1_unit, 0), np.append(v2_unit, 0))[2]
                            if cross < 0:
                                angle = -angle
                            curvature[i] = angle

        # Smooth curvature using a Gaussian filter
        sigma = max(1, len(curvature) // 50)  # Scale sigma with track length
        curvature = gaussian_filter(curvature, sigma=sigma)

        # Find the track center for oval-specific adjustments
        track_center_x = track_map.shape[1] / 2
        track_center_y = track_map.shape[0] / 2
        track_center = np.array([track_center_x, track_center_y])

        # Get binary version where track = 1, everything else = 0
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        dist_transform = distance_transform_edt(binary_track)

        # Create adjusted racing line by shifting points based on curvature
        adjusted_line = np.copy(centerline)
        
        # For oval tracks, use larger shifts in turns
        max_shift_factor = 0.6  # Increased from 0.4 for more aggressive cornering
        
        # Identify straights and turns
        curvature_threshold = 0.005  # Lower threshold to better identify turns in ovals
        is_turning = np.abs(curvature) > curvature_threshold
        
        # Find continuous segments (turns and straights)
        segments = []
        current_type = is_turning[0]
        segment_start = 0
        
        for i in range(1, len(is_turning)):
            if is_turning[i] != current_type:
                segments.append((segment_start, i-1, current_type))
                segment_start = i
                current_type = is_turning[i]
        
        # Add the last segment
        segments.append((segment_start, len(is_turning)-1, current_type))
        
        # Process each segment differently based on whether it's a turn or straight
        for start, end, is_turn in segments:
            for i in range(start, end+1):
                point = centerline[i].astype(int)
                if (0 <= point[0] < track_map.shape[1] and 
                    0 <= point[1] < track_map.shape[0]):
                    
                    # Get vector from track center to current point
                    vec_to_point = point - track_center
                    
                    # Get local track width from distance transform
                    local_width = dist_transform[point[1], point[0]] * 2
                    
                    if is_turn:
                        # In turns: shift more toward the inside (apex)
                        # Higher curvature = larger shift toward inside of curve
                        raw_shift = max_shift_factor * local_width * curvature[i]
                        
                        # For ovals, we want a more aggressive cut-in on turns
                        shift_amount = np.clip(raw_shift, -local_width*0.6, local_width*0.6)
                    else:
                        # In straights: stay more to the outside
                        # For oval tracks, this means staying farther from the center
                        dist_to_center = np.linalg.norm(vec_to_point)
                        
                        # Direction vector pointing away from center (normalized)
                        if dist_to_center > 0:
                            dir_from_center = vec_to_point / dist_to_center
                            
                            # Slight push outward on straights
                            shift_amount = local_width * 0.2
                            normal = dir_from_center  # Use direction from center as the normal
                            
                            # Apply the shift
                            adjusted_line[i] = centerline[i] + normal * shift_amount
                            continue  # Skip the normal calculation below

                    # For turns, calculate normal vector (perpendicular to path direction)
                    if i < len(centerline) - 1:
                        tangent = centerline[(i+1) % len(centerline)] - centerline[i]
                    else:
                        tangent = centerline[i] - centerline[(i-1) % len(centerline)]
                        
                    norm = np.linalg.norm(tangent)
                    if norm > 0:
                        tangent = tangent / norm
                        normal = np.array([-tangent[1], tangent[0]])  # Perpendicular vector
                        
                        # Shift point along normal vector
                        adjusted_line[i] = centerline[i] + normal * shift_amount

        # Final smoothing pass specifically for oval tracks
        # Use a more aggressive smoothing parameter
        try:
            adjusted_line = self.smooth_path(adjusted_line, smoothness=15)
        except:
            # If smoothing fails, use the original adjusted line
            pass

        # Verify all points are still on track
        for i in range(len(adjusted_line)):
            x, y = int(adjusted_line[i, 0]), int(adjusted_line[i, 1])
            if (x < 0 or x >= track_map.shape[1] or 
                y < 0 or y >= track_map.shape[0] or
                track_map[y, x] != 1):
                # If point is off track, revert to centerline
                adjusted_line[i] = centerline[i]

        return adjusted_line
        
    def calculate_speed_profile(self, racing_line):
        """
        Calculate optimal speed profile for the racing line using a three-step process:
        1. Calculate maximum speed at each point based on curvature
        2. Forward pass to limit acceleration
        3. Backward pass to limit deceleration
        """
        if len(racing_line) < 2:
            return []
            
        # Car parameters
        max_v = self.car_params['max_velocity']
        max_a = self.car_params['max_acceleration']
        max_d = self.car_params['max_deceleration']
        min_r = self.car_params['min_turn_radius']
        
        # Initialize arrays for calculations
        num_points = len(racing_line)
        curvature = np.zeros(num_points)
        speeds = np.zeros(num_points)
        distances = np.zeros(num_points)
        
        # Calculate distance between consecutive points
        for i in range(1, num_points):
            distances[i] = np.linalg.norm(racing_line[i] - racing_line[i-1])
        
        # Calculate curvature at each point using a sliding window approach
        window_size = min(7, num_points // 10)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        
        half_window = window_size // 2
        
        for i in range(num_points):
            # Get indices for window, handling wrap-around for closed loop
            indices = [(i - half_window + j) % num_points for j in range(window_size)]
            window_points = racing_line[indices]
            
            # Calculate curvature using circle fitting
            if len(window_points) >= 3:
                try:
                    x = window_points[:, 0]
                    y = window_points[:, 1]
                    
                    # Center the points
                    mean_x = np.mean(x)
                    mean_y = np.mean(y)
                    x_centered = x - mean_x
                    y_centered = y - mean_y
                    
                    # Solve for circle parameters
                    A = np.column_stack([x_centered, y_centered, np.ones(len(x))])
                    b = -x_centered**2 - y_centered**2
                    
                    # Use least squares to solve for circle parameters
                    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    
                    # Calculate radius
                    radius = np.sqrt((params[0]/2)**2 + (params[1]/2)**2 - params[2])
                    
                    # Curvature = 1/radius (with some minimum to avoid infinity)
                    curvature[i] = 1.0 / max(radius, 0.001)
                except:
                    # Fallback to simpler angle-based curvature estimation
                    prev_idx = (i - 1) % num_points
                    next_idx = (i + 1) % num_points
                    
                    prev = racing_line[prev_idx]
                    curr = racing_line[i]
                    next_pt = racing_line[next_idx]
                    
                    v1 = curr - prev
                    v2 = next_pt - curr
                    
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    
                    if v1_norm > 0 and v2_norm > 0:
                        v1 = v1 / v1_norm
                        v2 = v2 / v2_norm
                        
                        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                        angle_change = np.arccos(dot_product)
                        
                        # Estimate radius based on angle and distance
                        segment_length = (v1_norm + v2_norm) / 2
                        if angle_change > 0.001:
                            radius = segment_length / angle_change
                            curvature[i] = 1.0 / max(radius, 0.001)
        
        # Smooth the curvature
        sigma = max(1, num_points // 50)
        curvature = gaussian_filter(curvature, sigma=sigma)
        
        # STEP 1: Calculate max speed based on curvature (lateral acceleration limit)
        max_lateral_accel = 2.0  # m/s^2 (adjust based on car capabilities)
        
        for i in range(num_points):
            # Convert pixels to meters if applicable
            # Here we assume a simple conversion factor - adjust as needed
            pixel_to_meter = 0.01  # Example: 1 pixel = 1 cm
            
            # Calculate max speed based on curvature and lateral acceleration limit
            # v_max = sqrt(a_lat * r) where r = 1/curvature
            if curvature[i] > 0.001:  # For curved sections
                v_max_curve = min(max_v, np.sqrt(max_lateral_accel / curvature[i]))
                speeds[i] = v_max_curve
            else:  # For straight sections
                speeds[i] = max_v
        
        # STEP 2: Forward pass - constraint by acceleration
        # Start with a reasonable initial speed
        speeds[0] = max(0.5 * max_v, min(max_v, speeds[0]))
        
        for i in range(1, num_points):
            # Current max speed due to acceleration constraint
            # v^2 = v_0^2 + 2*a*s
            v_prev = speeds[i-1]
            dist = distances[i] * pixel_to_meter  # Convert to meters
            
            v_max_accel = np.sqrt(v_prev**2 + 2 * max_a * dist)
            speeds[i] = min(speeds[i], v_max_accel)
            
        # STEP 3: Backward pass - constraint by deceleration
        for i in range(num_points-2, -1, -1):
            v_next = speeds[(i+1) % num_points]
            dist = distances[(i+1) % num_points] * pixel_to_meter  # Convert to meters
            
            v_max_decel = np.sqrt(v_next**2 + 2 * max_d * dist)
            speeds[i] = min(speeds[i], v_max_decel)
        
        # Final smoothing pass
        speeds = gaussian_filter(speeds, sigma=sigma/2)
        
        return speeds

    def _evaluate_racing_line(self, racing_line, track_map):
        """
        Evaluate a racing line based on multiple criteria:
        1. Time to complete the track (calculated from speed profile)
        2. Smoothness of the racing line
        3. Penalty for getting close to track edges
        4. Reward for proper oval racing trajectory (with emphasis on straight-line speed)
        
        Returns a fitness score (higher is better)
        """
        if len(racing_line) < 2:
            return float('-inf')
        
        # Get distance transform for edge distance calculation
        binary_track = np.zeros_like(track_map)
        binary_track[track_map == 1] = 1
        dist_transform = distance_transform_edt(binary_track)
        
        # Calculate curvature and identify track segments
        curvature = np.zeros(len(racing_line))
        for i in range(len(racing_line)):
            if i > 0 and i < len(racing_line) - 1:
                prev = racing_line[i-1]
                curr = racing_line[i]
                next_pt = racing_line[i+1]
                v1 = curr - prev
                v2 = next_pt - curr
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                if v1_norm > 0 and v2_norm > 0:
                    v1 = v1 / v1_norm
                    v2 = v2 / v2_norm
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle_change = np.arccos(dot_product)
                    
                    # Determine if we're turning left or right
                    cross = np.cross(np.append(v1, 0), np.append(v2, 0))[2]
                    if cross < 0:
                        angle_change = -angle_change
                    
                    curvature[i] = angle_change
        
        # Smooth the curvature
        sigma = max(1, len(curvature) // 50)
        curvature = gaussian_filter(curvature, sigma=sigma)
        
        # Calculate distance between consecutive points
        distances = np.zeros(len(racing_line))
        for i in range(1, len(racing_line)):
            distances[i] = np.linalg.norm(racing_line[i] - racing_line[i-1])
        total_distance = np.sum(distances)
        
        # Calculate speed profile
        max_lateral_accel = 2.0  # m/s^2
        max_v = self.car_params['max_velocity']
        pixel_to_meter = 0.01  # Convert pixels to meters
        
        speeds = np.zeros(len(racing_line))
        for i in range(len(racing_line)):
            if abs(curvature[i]) > 0.001:  # Check absolute value for both left and right turns
                v_max_curve = min(max_v, np.sqrt(max_lateral_accel / max(abs(curvature[i]), 0.001)))
                speeds[i] = v_max_curve
            else:
                speeds[i] = max_v
        
        # Calculate time to complete the track
        time_segments = distances[1:] * pixel_to_meter / np.maximum(speeds[1:], 0.01)
        total_time = np.sum(time_segments)
        
        # Calculate straightness of segments (higher is better)
        # Straightness is the average of the dot products between consecutive segments
        straightness = 0.0
        for i in range(len(racing_line)-2):
            v1 = racing_line[i+1] - racing_line[i]
            v2 = racing_line[i+2] - racing_line[i+1]
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm > 0 and v2_norm > 0:
                dot_product = np.dot(v1/v1_norm, v2/v2_norm)
                straightness += dot_product
        
        # Normalize straightness
        if len(racing_line) > 2:
            straightness /= (len(racing_line) - 2)
            # Scale to be more impactful (1 is perfectly straight, -1 is zigzag)
            straightness = (straightness + 1) / 2  # Rescale from [-1,1] to [0,1]
            straightness = straightness ** 2  # Square to emphasize straight sections
        else:
            straightness = 0
        
        # Calculate smoothness (lower curvature changes are better)
        curvature_changes = np.diff(curvature, append=curvature[0])
        smoothness = -np.sum(curvature_changes**2)
        
        # Calculate distance from track edges - IMPROVED BOUNDARY CHECKING
        edge_distances = np.zeros(len(racing_line))
        off_track_count = 0
        
        for i in range(len(racing_line)):
            x, y = int(racing_line[i, 0]), int(racing_line[i, 1])
            if 0 <= x < track_map.shape[1] and 0 <= y < track_map.shape[0]:
                if track_map[y, x] == 1:
                    edge_distances[i] = dist_transform[y, x]
                else:
                    # Very high penalty for off-track points
                    edge_distances[i] = 0.01  # Nearly zero
                    off_track_count += 1
            else:
                edge_distances[i] = 0.01  # Nearly zero for out of bounds
                off_track_count += 1
        
        # Strong penalty for any off-track points - makes this a hard constraint
        off_track_penalty = -100.0 * off_track_count if off_track_count > 0 else 0.0
        
        # Standard edge penalty (becomes more important when all points are on track)
        edge_penalty = -np.sum(1.0 / np.maximum(edge_distances, 0.1))
        
        # Specific oval track strategy bonus - MODIFIED FOR STRAIGHTER RACING LINES:
        oval_strategy_bonus = 0.0
        
        # Identify straights and turns
        curvature_threshold = 0.01  # Threshold for detecting turns vs straights
        is_turning = np.abs(curvature) > curvature_threshold
        
        # Find continuous segments of turns and straights
        segments = []
        current_type = is_turning[0]
        segment_start = 0
        
        for i in range(1, len(is_turning)):
            if is_turning[i] != current_type:
                segments.append((segment_start, i-1, current_type))
                segment_start = i
                current_type = is_turning[i]
        
        # Add the last segment
        segments.append((segment_start, len(is_turning)-1, current_type))
        
        # Track center for reference
        center_x = track_map.shape[1] / 2
        center_y = track_map.shape[0] / 2
        
        # For each segment, calculate strategy bonus
        for start, end, is_turn in segments:
            segment_points = racing_line[start:end+1]
            
            if is_turn:
                # For turns: Use a modified approach that balances aggressive turning
                # with staying on track
                
                # Calculate direction of turn (using average curvature in segment)
                avg_curvature = np.mean(curvature[start:end+1])
                turn_direction = -1 if avg_curvature < 0 else 1
                
                # Find the optimal apex point (maximum curvature point)
                apex_idx = start + np.argmax(np.abs(curvature[start:end+1]))
                
                for i in range(start, end+1):
                    point = racing_line[i]
                    
                    # Distance from point to center
                    vec_to_center = point - np.array([center_x, center_y])
                    dist_to_center = np.linalg.norm(vec_to_center)
                    
                    # Distance from apex affects how much we cut the corner
                    dist_to_apex = abs(i - apex_idx) / max(1, end - start)
                    
                    # We want to cut more at the apex, less at entry/exit
                    # for smoother transition to straights
                    if dist_to_apex < 0.3:  # Near apex
                        # Reward cutting in more aggressively at apex
                        oval_strategy_bonus -= dist_to_center * 0.002
                    else:
                        # Less aggressive cutting away from apex for smoother entry/exit
                        oval_strategy_bonus -= dist_to_center * 0.0005
            else:
                # For straights: heavily reward staying straight (minimal direction changes)
                # and maintaining a consistent path
                
                # First, calculate straightness of this specific straight segment
                if len(segment_points) > 2:
                    segment_straightness = 0.0
                    
                    # Calculate average direction vector for the entire straight
                    start_pt = segment_points[0]
                    end_pt = segment_points[-1]
                    main_direction = end_pt - start_pt
                    if np.linalg.norm(main_direction) > 0:
                        main_direction = main_direction / np.linalg.norm(main_direction)
                        
                        # Reward points that follow this main direction
                        for j in range(len(segment_points)-1):
                            segment_vec = segment_points[j+1] - segment_points[j]
                            if np.linalg.norm(segment_vec) > 0:
                                segment_vec = segment_vec / np.linalg.norm(segment_vec)
                                # How well does this segment align with main direction?
                                alignment = np.dot(segment_vec, main_direction)
                                # Heavy reward for maintaining straight line
                                segment_straightness += alignment ** 2  # Square to emphasize straightness
                    
                    # Normalize and add bonus
                    segment_straightness /= max(1, len(segment_points)-1)
                    oval_strategy_bonus += segment_straightness * 20.0  # High weight for straightness
                
                # Also slightly reward staying farther from center on straights
                # This encourages using the full width of the track on straights
                for i in range(start, end+1):
                    point = racing_line[i]
                    vec_to_center = point - np.array([center_x, center_y])
                    dist_to_center = np.linalg.norm(vec_to_center)
                    oval_strategy_bonus += dist_to_center * 0.0005
        
        # Combine the factors with weights - ADJUSTED FOR STRAIGHT-LINE EMPHASIS
        w_time = 1.0            # Time is still most important
        w_smoothness = 0.5      # Smooth transitions
        w_edge = 0.5            # Increased penalty for edge proximity
        w_straight = 5.0        # Very high weight for straightness
        w_oval_strategy = 1.0   # Slightly increased for oval strategy
        
        # Add a special bonus for staying on track (hard constraint)
        on_track_bonus = 500.0 if off_track_count == 0 else 0.0
        
        # Fitness (higher is better)
        fitness = (-w_time * total_time + 
                  w_smoothness * smoothness - 
                  w_edge * edge_penalty + 
                  w_straight * straightness * 100 +  # Scale up to make more impactful
                  w_oval_strategy * oval_strategy_bonus +
                  on_track_bonus +
                  off_track_penalty)  # Strong penalty for off-track points
        
        return fitness

    def _robust_smooth_path(self, points, smoothness=30):
        """
        A more robust path smoothing algorithm that handles challenging point sets
        that might cause splprep to fail.
        """
        if len(points) < 4:
            return points
            
        # Convert points to numpy array if not already
        points = np.array(points)
        
        # Check if points are valid (no NaN, no duplicates)
        # Remove duplicates
        valid_points = []
        for i in range(len(points)):
            if i == 0 or not np.array_equal(points[i], points[i-1]):
                if not np.isnan(points[i][0]) and not np.isnan(points[i][1]):
                    valid_points.append(points[i])
                    
        if len(valid_points) < 4:
            print("Not enough valid points for smoothing")
            return points
            
        valid_points = np.array(valid_points)
        
        # Try a simple moving average first
        window_size = min(5, len(valid_points) // 5)
        window_size = max(3, window_size) # At least 3
        
        smoothed = np.copy(valid_points)
        # Apply moving average with wrap-around for closed loops
        for i in range(len(valid_points)):
            window_indices = [(i - window_size//2 + j) % len(valid_points) for j in range(window_size)]
            smoothed[i] = np.mean(valid_points[window_indices], axis=0)
            
        # If we have enough points, try a more advanced smoothing
        if len(smoothed) >= 10:
            try:
                # Resample to have uniform point distribution
                cumulative_distances = np.zeros(len(smoothed))
                for i in range(1, len(smoothed)):
                    cumulative_distances[i] = cumulative_distances[i-1] + np.linalg.norm(smoothed[i] - smoothed[i-1])
                
                # Create new parameter array with uniform spacing
                total_distance = cumulative_distances[-1]
                desired_points = min(500, len(smoothed) * 2)
                uniform_distances = np.linspace(0, total_distance, desired_points)
                
                # Interpolate x and y coordinates separately
                x_interp = np.interp(uniform_distances, cumulative_distances, smoothed[:, 0])
                y_interp = np.interp(uniform_distances, cumulative_distances, smoothed[:, 1])
                
                # Combine into final smooth path
                resampled = np.column_stack((x_interp, y_interp))
                
                # Apply one more round of smoothing with Gaussian filter
                sigma = max(1, len(resampled) // 50)
                final_smooth_x = gaussian_filter(resampled[:, 0], sigma=sigma)
                final_smooth_y = gaussian_filter(resampled[:, 1], sigma=sigma)
                
                return np.column_stack((final_smooth_x, final_smooth_y))
            except Exception as e:
                print(f"Advanced smoothing failed: {e}")
                return smoothed
        else:
            return smoothed