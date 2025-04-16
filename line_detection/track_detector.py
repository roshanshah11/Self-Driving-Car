#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Detector for Limelight data processing
Processes line data from Limelight and builds a track map
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

class TrackDetector:
    """
    Track Detector class responsible for processing line data and building a track map.
    
    This class handles:
    1. Taking line data from Limelight and building boundary representations
    2. Maintaining a map of the track as the robot moves
    3. Visualizing the track with boundaries, robot path, and optionally racing lines
    """
    def __init__(self, map_size=(800, 800), cell_size=10):
        # Map configuration
        self.map_size = map_size  # Size of the mapping grid in cells
        self.cell_size = cell_size  # Size of each cell in pixels
        
        # Initialize empty track map (0 = unknown, 1 = track, 2 = boundary)
        self.track_map = np.zeros(map_size, dtype=np.uint8)
        
        # Track boundaries
        self.inner_boundary = []
        self.outer_boundary = []
        
        # Robot positioning
        self.robot_position = (map_size[0]//2, map_size[1]//2)  # Start in the middle
        self.robot_orientation = 0  # 0 degrees (facing "north")
        
        # Threshold parameters
        self.use_adaptive_threshold = True
        self.adaptive_block_size = 11  # Must be odd
        self.adaptive_constant = 2
        self.white_threshold = 180  # Default threshold for white pixels
        
        # History tracking
        self.prev_inner_boundary = None
        self.prev_outer_boundary = None
        self.boundary_history = []
        self.history_size = 5
        
        # Racing line (optional)
        self.optimal_line = None
        
        # Path tracking
        self.position_history = []
        self.max_history_points = 1000
        
        # Track completion
        self.track_complete = False
        self.minimal_distance_for_completion = 50  # Pixels distance to determine track completion
        self.min_track_length = 200  # Minimum track length before considering completion
        
        # Visualization
        self.latest_inner_points = None
        self.latest_outer_points = None
    
    # -------------------------------------------------
    # Line Data Processing Methods
    # -------------------------------------------------
    
    def update_from_line_data(self, left_line, right_line, robot_pose):
        """
        Update track map using direct line data from Limelight
        
        Args:
            left_line: [x1, y1, x2, y2] for left boundary line
            right_line: [x1, y1, x2, y2] for right boundary line
            robot_pose: (x, y, theta) robot position and orientation
        
        Returns:
            Tuple of (inner_boundary, outer_boundary) representing the detected boundaries
        """
        # Skip if no line data is available
        if left_line is None and right_line is None:
            print("No line data available")
            return None, None

        # Update robot position
        self.robot_position = (int(robot_pose[0]), int(robot_pose[1]))
        self.robot_orientation = robot_pose[2]

        # Add to position history
        self.position_history.append(self.robot_position)
        if len(self.position_history) > self.max_history_points:
            self.position_history.pop(0)

        # Check if track is complete (robot has returned to starting area)
        self._check_track_completion()

        # Convert line data to boundaries
        inner_boundary, outer_boundary = self._lines_to_boundaries(left_line, right_line)
        
        # Update the map with the new boundaries
        self.update_map(inner_boundary, outer_boundary, robot_pose)

        # Return the boundaries for visualization
        return inner_boundary, outer_boundary
    
    def _lines_to_boundaries(self, left_line, right_line):
        """
        Convert line data [x1, y1, x2, y2] to boundary contours
        
        Args:
            left_line: [x1, y1, x2, y2] for left line
            right_line: [x1, y1, x2, y2] for right line
            
        Returns:
            Tuple of (inner_boundary, outer_boundary) arrays in contour format
        """
        def line_to_contour(line_points):
            if line_points is None:
                return None

            x1, y1, x2, y2 = line_points

            # Create a line with multiple points (not just endpoints)
            # This helps with mapping and visualization
            num_points = 10
            points = []

            for i in range(num_points):
                t = i / (num_points - 1)
                x = int(x1 + t * (x2 - x1))
                y = int(y1 + t * (y2 - y1))
                points.append([[x, y]])

            return np.array(points, dtype=np.int32)

        # Convert lines to contour format
        inner_boundary = line_to_contour(left_line)
        outer_boundary = line_to_contour(right_line)

        # Store as previous boundaries for continuity
        if inner_boundary is not None:
            self.prev_inner_boundary = inner_boundary
        if outer_boundary is not None:
            self.prev_outer_boundary = outer_boundary

        return inner_boundary, outer_boundary
    
    # -------------------------------------------------
    # Map Update Methods
    # -------------------------------------------------
    
    def update_map(self, inner_boundary, outer_boundary, robot_pose, transform_coords=True):
        """
        Update the track map with new boundary information
        
        Args:
            inner_boundary: Inner boundary contour
            outer_boundary: Outer boundary contour
            robot_pose: (x, y, theta) robot position and orientation
            transform_coords: Whether to transform coordinates from robot to map
        """
        if inner_boundary is None and outer_boundary is None:
            print("WARNING: No boundary data for map update")
            return
            
        # Update robot position
        self.robot_position = (int(robot_pose[0]), int(robot_pose[1]))
        self.robot_orientation = robot_pose[2]
        
        # Transform boundary points to map coordinates if needed
        inner_map_points, outer_map_points = self._transform_to_map_coords(
            inner_boundary, outer_boundary, transform_coords
        )
        
        # Log debug information
        self._log_boundary_stats(inner_map_points, outer_map_points)
        
        # Draw thicker lines for better visibility in the map
        line_thickness = 2
        
        # Update the map with the new boundaries
        self._draw_boundaries_on_map(inner_map_points, outer_map_points, line_thickness)
        
        # Fill the track area between boundaries
        self._fill_track_area(inner_map_points, outer_map_points)
        
        # Store latest boundary points for visualization
        self._update_latest_boundaries(inner_map_points, outer_map_points)
    
    def _transform_to_map_coords(self, inner_boundary, outer_boundary, transform_coords):
        """
        Transform boundary points from robot-relative to map coordinates
        
        Args:
            inner_boundary: Inner boundary contour
            outer_boundary: Outer boundary contour
            transform_coords: Whether to transform coordinates
            
        Returns:
            Tuple of (inner_map_points, outer_map_points) in map coordinates
        """
        if transform_coords:
            inner_map_points = self._world_to_map(inner_boundary) if inner_boundary is not None else np.array([])
            outer_map_points = self._world_to_map(outer_boundary) if outer_boundary is not None else np.array([])
        else:
            # Use points directly if already in map coordinates
            inner_map_points = np.array(inner_boundary).reshape(-1, 2) if inner_boundary is not None else np.array([])
            outer_map_points = np.array(outer_boundary).reshape(-1, 2) if outer_boundary is not None else np.array([])
            
        return inner_map_points, outer_map_points
    
    def _world_to_map(self, points):
        """
        Convert from robot-relative to map coordinates
        
        Args:
            points: Array of points in robot-relative coordinates
            
        Returns:
            Array of points in map coordinates
        """
        x_offset, y_offset = self.robot_position
        transformed_points = []
        
        for point in points:
            # Extract point coordinates
            x = point[0][0]
            y = point[0][1]
            
            # Rotate by robot orientation
            cos_theta = np.cos(self.robot_orientation)
            sin_theta = np.sin(self.robot_orientation)
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta
            
            # Translate to robot position
            map_x = int(x_rot + x_offset)
            map_y = int(y_rot + y_offset)
            
            # Ensure points are within map bounds
            if 0 <= map_x < self.map_size[0] and 0 <= map_y < self.map_size[1]:
                transformed_points.append([map_x, map_y])
                
        return np.array(transformed_points)
    
    def _log_boundary_stats(self, inner_map_points, outer_map_points):
        """Log boundary statistics for debugging"""
        print(f"Robot position: {self.robot_position}, orientation: {self.robot_orientation:.2f}")
        print(f"Inner boundary points: {len(inner_map_points)}, Outer boundary points: {len(outer_map_points)}")
        
        if len(inner_map_points) > 0:
            print(f"First inner map point: {inner_map_points[0]}")
        if len(outer_map_points) > 0:
            print(f"First outer map point: {outer_map_points[0]}")
    
    def _draw_boundaries_on_map(self, inner_map_points, outer_map_points, line_thickness):
        """Draw boundaries on the track map"""
        if len(inner_map_points) > 0:
            try:
                cv2.polylines(self.track_map, [inner_map_points], isClosed=False, 
                              color=2, thickness=line_thickness)
                print("Drew inner boundary line on map")
            except Exception as e:
                print(f"ERROR drawing inner boundary: {e}")
        
        if len(outer_map_points) > 0:
            try:
                cv2.polylines(self.track_map, [outer_map_points], isClosed=False, 
                              color=2, thickness=line_thickness)
                print("Drew outer boundary line on map")
            except Exception as e:
                print(f"ERROR drawing outer boundary: {e}")
    
    def _fill_track_area(self, inner_map_points, outer_map_points):
        """Fill the track area between boundaries"""
        if len(inner_map_points) > 0 and len(outer_map_points) > 0:
            try:
                # Create a temporary mask for filling
                mask = np.zeros_like(self.track_map)
                cv2.fillPoly(mask, [outer_map_points], color=1)
                cv2.fillPoly(mask, [inner_map_points], color=0)
                
                # Update track map where mask is 1
                self.track_map = np.maximum(self.track_map, mask)
                print("Filled track area between boundaries")
                print(f"Track map has {np.sum(self.track_map > 0)} non-zero pixels")
            except Exception as e:
                print(f"ERROR filling track area: {e}")
    
    def _update_latest_boundaries(self, inner_map_points, outer_map_points):
        """Store latest boundary points for visualization"""
        if len(inner_map_points) > 0:
            self.latest_inner_points = inner_map_points
        
        if len(outer_map_points) > 0:
            self.latest_outer_points = outer_map_points
    
    def _check_track_completion(self):
        """Check if the track mapping is complete"""
        if len(self.position_history) > self.min_track_length:
            start_pos = self.position_history[0]
            current_pos = self.position_history[-1]
            distance_to_start = np.sqrt(
                (current_pos[0] - start_pos[0])**2 + 
                (current_pos[1] - start_pos[1])**2
            )

            if distance_to_start < self.minimal_distance_for_completion:
                self.track_complete = True
                print("Track mapping complete! Robot has returned to starting position.")
    
    # -------------------------------------------------
    # Visualization Methods
    # -------------------------------------------------
    
    def get_visualization(self):
        """
        Return a visualization of the track map for display
        
        Returns:
            RGB image of the track map
        """
        # Debug map content
        print(f"Generating visualization from track map with shape {self.track_map.shape}")
        num_unknown = np.sum(self.track_map == 0)
        num_track = np.sum(self.track_map == 1)
        num_boundary = np.sum(self.track_map == 2)
        print(f"Track map contains: {num_unknown} unknown, {num_track} track, {num_boundary} boundary pixels")
        
        # Create RGB visualization
        vis_map = np.zeros((self.map_size[0], self.map_size[1], 3), dtype=np.uint8)
        
        # Unknown areas (black)
        vis_map[self.track_map == 0] = [0, 0, 0]
        
        # Track areas (green)
        vis_map[self.track_map == 1] = [0, 200, 0]
        
        # Boundaries (white)
        vis_map[self.track_map == 2] = [255, 255, 255]
        
        # Draw robot position
        self._draw_robot_on_map(vis_map)
        
        # Add text legend
        self._add_legend_to_map(vis_map)
        
        # Add grid lines
        self._add_grid_to_map(vis_map)
        
        # Center reference point (always visible)
        center_x, center_y = self.map_size[0] // 2, self.map_size[1] // 2
        cv2.circle(vis_map, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.putText(vis_map, "CENTER", (center_x+10, center_y+5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        return vis_map
    
    def _draw_robot_on_map(self, vis_map):
        """Draw robot position and direction on the map"""
        robot_x, robot_y = self.robot_position
        if 0 <= robot_x < self.map_size[0] and 0 <= robot_y < self.map_size[1]:
            # Draw red dot for robot
            cv2.circle(vis_map, (robot_x, robot_y), 5, (255, 0, 0), -1)
            
            # Draw direction indicator
            direction_len = 20
            end_x = int(robot_x + direction_len * np.cos(self.robot_orientation))
            end_y = int(robot_y + direction_len * np.sin(self.robot_orientation))
            cv2.line(vis_map, (robot_x, robot_y), (end_x, end_y), (255, 0, 0), 2)
    
    def _add_legend_to_map(self, vis_map):
        """Add text legend to map visualization"""
        legend_y = 30
        cv2.putText(vis_map, "Track Map", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(vis_map, "- White: Track Boundaries", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.putText(vis_map, "- Green: Track Surface", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        legend_y += 20
        cv2.putText(vis_map, "- Red: Robot Position", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    def _add_grid_to_map(self, vis_map):
        """Add coordinate grid to map visualization"""
        grid_color = (30, 30, 30)  # Dark gray
        grid_spacing = 100
        
        # Draw grid lines
        for x in range(0, self.map_size[0], grid_spacing):
            cv2.line(vis_map, (x, 0), (x, self.map_size[1]-1), grid_color, 1)
        for y in range(0, self.map_size[1], grid_spacing):
            cv2.line(vis_map, (0, y), (self.map_size[0]-1, y), grid_color, 1)
            
        # Add coordinates at grid intersections
        font_scale = 0.3
        for x in range(0, self.map_size[0], grid_spacing):
            for y in range(0, self.map_size[1], grid_spacing):
                cv2.putText(vis_map, f"{x},{y}", (x+5, y+15), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (100, 100, 100), 1)
    
    def visualize_track(self, show_optimal_line=False, optimal_line=None, speed_profile=None):
        """
        Visualize the track with optional racing line and speed profile
        
        Args:
            show_optimal_line: Whether to show the optimal racing line
            optimal_line: The optimal racing line to draw (if not provided, self.optimal_line is used)
            speed_profile: Speed profile for the racing line (optional)
            
        Returns:
            The visualization image
        """
        # Start with the base visualization
        vis_map = self.get_visualization()
        
        # Draw car position history as a trail
        self._draw_position_history(vis_map)
        
        # Add track completion status
        if self.track_complete:
            cv2.putText(vis_map, "TRACK MAPPING COMPLETE", (10, self.map_size[1] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add optimal racing line if requested
        if show_optimal_line:
            self._draw_racing_line(vis_map, optimal_line, speed_profile)
        
        return vis_map
    
    def _draw_position_history(self, vis_map):
        """Draw the robot's path on the map"""
        if len(self.position_history) > 1:
            # Convert position history to numpy array for drawing
            history_array = np.array(self.position_history, dtype=np.int32)
            
            # Draw path with gradient from blue to cyan
            for i in range(len(history_array) - 1):
                intensity = int(255 * (i / len(history_array)))
                color = (intensity, intensity, 255)  # Gradient from blue to cyan
                cv2.line(vis_map, tuple(history_array[i]), tuple(history_array[i+1]), color, 2)
            
            # Highlight start position
            if len(history_array) > 0:
                cv2.circle(vis_map, tuple(history_array[0]), 8, (0, 255, 255), -1)  # Yellow circle
                cv2.putText(vis_map, "START", (history_array[0][0] + 10, history_array[0][1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Add legend entries
        cv2.putText(vis_map, "- Blue: Car Path", (10, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
        cv2.putText(vis_map, "- Yellow: Start Position", (10, 180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _draw_racing_line(self, vis_map, optimal_line=None, speed_profile=None):
        """Draw the optimal racing line on the map"""
        # Use provided optimal line or instance attribute
        line_to_draw = optimal_line if optimal_line is not None else self.optimal_line
        
        if line_to_draw is not None and len(line_to_draw) > 0:
            # Prepare racing line points
            racing_line_points = self._prepare_racing_line_points(line_to_draw)
            
            # Close the loop if needed
            racing_line_points = self._close_racing_line_loop(racing_line_points, speed_profile)
            
            # Draw the racing line (with or without speed profile)
            if speed_profile is not None and len(speed_profile) > 0:
                self._draw_racing_line_with_speed(vis_map, racing_line_points, speed_profile)
            else:
                self._draw_simple_racing_line(vis_map, racing_line_points)
    
    def _prepare_racing_line_points(self, line_to_draw):
        """Prepare racing line points for drawing"""
        # Convert to numpy array if not already
        if not isinstance(line_to_draw, np.ndarray):
            racing_line_points = np.array(line_to_draw, dtype=np.int32)
        else:
            racing_line_points = line_to_draw.copy()
        
        # Reshape if needed to ensure we have a 2D array of points
        if len(racing_line_points.shape) == 3 and racing_line_points.shape[1] == 1:
            racing_line_points = racing_line_points.reshape(-1, 2)
        elif len(racing_line_points.shape) == 1 and racing_line_points.shape[0] % 2 == 0:
            # If it's a flat array of [x1, y1, x2, y2, ...], reshape to [[x1, y1], [x2, y2], ...]
            racing_line_points = racing_line_points.reshape(-1, 2)
        
        # Ensure points are integers
        racing_line_points = np.array(racing_line_points, dtype=np.int32).reshape(-1, 2)
        
        # Print debug info
        if len(racing_line_points) > 0:
            print(f"Racing line has {len(racing_line_points)} points")
            print(f"First: {racing_line_points[0]}, last: {racing_line_points[-1]}")
        
        return racing_line_points
    
    def _close_racing_line_loop(self, racing_line_points, speed_profile=None):
        """Close the racing line loop if first and last points aren't close"""
        if len(racing_line_points) > 2:
            # Check if the racing line is already closed
            first_point = racing_line_points[0]
            last_point = racing_line_points[-1]
            dist = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
            
            # If distance is greater than 30 pixels, consider it not closed
            if dist > 30:
                # Add the first point to the end to close the loop
                racing_line_points = np.vstack([racing_line_points, racing_line_points[0:1]])
                
                # If we have a speed profile, extend it with the first value to match
                if speed_profile is not None:
                    speed_profile = np.append(speed_profile, speed_profile[0])
        
        return racing_line_points
    
    def _draw_racing_line_with_speed(self, vis_map, racing_line_points, speed_profile):
        """Draw the racing line with speed-based coloring"""
        # Ensure speed profile matches racing line length
        if len(speed_profile) != len(racing_line_points):
            print(f"WARNING: Speed profile length ({len(speed_profile)}) doesn't match racing line points ({len(racing_line_points)})")
            if len(speed_profile) > 1:
                indices = np.linspace(0, len(speed_profile)-1, len(racing_line_points))
                speed_profile = np.interp(indices, np.arange(len(speed_profile)), speed_profile)
        
        # Normalize speeds for color mapping
        max_speed = np.max(speed_profile)
        min_speed = np.min(speed_profile)
        speed_range = max_speed - min_speed
        
        # Add speed profile legend
        self._add_speed_legend(vis_map, min_speed, max_speed)
        
        # Draw racing line with speed-based colors
        line_thickness = 4
        for i in range(len(racing_line_points) - 1):
            # Normalize speed for this segment
            if speed_range > 0:
                t = (speed_profile[i] - min_speed) / speed_range
            else:
                t = 0.5
                
            # Color based on speed: blue (slow) to red (fast)
            b = int(255 * (1 - t))
            r = int(255 * t)
            g = 0
            
            # Draw line segment
            pt1 = (int(racing_line_points[i][0]), int(racing_line_points[i][1]))
            pt2 = (int(racing_line_points[i+1][0]), int(racing_line_points[i+1][1]))
            cv2.line(vis_map, pt1, pt2, (b, g, r), line_thickness)
        
        # Add speed markers
        self._add_speed_markers(vis_map, racing_line_points, speed_profile, min_speed, max_speed, speed_range)
        
        # Add legend entry
        cv2.putText(vis_map, "- Racing Line (color = speed)", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 0), 1)
    
    def _add_speed_legend(self, vis_map, min_speed, max_speed):
        """Add speed gradient legend to the visualization"""
        cv2.putText(vis_map, "- Speed Profile:", (10, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add color gradient legend for speed
        legend_width = 100
        legend_height = 15
        legend_x = 30
        legend_y = 210
        
        # Draw gradient
        for i in range(legend_width):
            t = i / legend_width
            b = int(255 * (1 - t))
            r = int(255 * t)
            cv2.line(vis_map, (legend_x + i, legend_y), 
                    (legend_x + i, legend_y + legend_height), (b, 0, r), 1)
        
        # Add min and max speed labels
        cv2.putText(vis_map, f"{min_speed:.1f}", (legend_x - 5, legend_y + legend_height + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
        cv2.putText(vis_map, f"{max_speed:.1f}", (legend_x + legend_width - 5, legend_y + legend_height + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 255), 1)
    
    def _add_speed_markers(self, vis_map, racing_line_points, speed_profile, min_speed, max_speed, speed_range):
        """Add speed indicator markers along the racing line"""
        # Limit number of markers to avoid clutter
        num_markers = min(20, len(racing_line_points))
        marker_indices = np.linspace(0, len(racing_line_points)-2, num_markers, dtype=int)
        
        for idx in marker_indices:
            # Get marker position and speed
            marker_pos = racing_line_points[idx]
            speed_val = speed_profile[idx]
            
            # Normalize speed for this marker
            if speed_range > 0:
                t = (speed_val - min_speed) / speed_range
            else:
                t = 0.5
            
            # Color based on speed
            b = int(255 * (1 - t))
            r = int(255 * t)
            g = 0
            
            # Draw marker
            marker_pos_tuple = (int(marker_pos[0]), int(marker_pos[1]))
            cv2.circle(vis_map, marker_pos_tuple, 6, (b, g, r), -1)
            cv2.circle(vis_map, marker_pos_tuple, 6, (255, 255, 255), 1)  # White outline
            
            # Add speed text
            cv2.putText(vis_map, f"{speed_val:.1f}", 
                        (marker_pos_tuple[0] + 8, marker_pos_tuple[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_simple_racing_line(self, vis_map, racing_line_points):
        """Draw the racing line without speed information"""
        for i in range(len(racing_line_points) - 1):
            try:
                pt1 = (int(racing_line_points[i][0]), int(racing_line_points[i][1]))
                pt2 = (int(racing_line_points[i+1][0]), int(racing_line_points[i+1][1]))
                cv2.line(vis_map, pt1, pt2, (255, 165, 0), 4)  # Orange color
            except (IndexError, TypeError) as e:
                print(f"Error drawing line segment {i}: {e}")
                continue
        
        # Add legend entry
        cv2.putText(vis_map, "- Orange: Racing Line", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # -------------------------------------------------
    # Utility Methods
    # -------------------------------------------------
    
    def get_distance_to_start(self):
        """
        Get distance from current position to start position
        
        Returns:
            Distance in pixels
        """
        if len(self.position_history) > 1:
            start_pos = self.position_history[0]
            current_pos = self.position_history[-1]
            return np.sqrt(
                (current_pos[0] - start_pos[0])**2 + 
                (current_pos[1] - start_pos[1])**2
            )
        return 0
        
    def get_distance_traveled(self):
        """
        Get total distance traveled along the path
        
        Returns:
            Distance in pixels
        """
        if len(self.position_history) > 1:
            total_dist = 0
            for i in range(len(self.position_history) - 1):
                p1 = self.position_history[i]
                p2 = self.position_history[i+1]
                total_dist += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            return total_dist
        return 0
    
    def is_track_complete(self):
        """Check if track mapping is complete"""
        return self.track_complete