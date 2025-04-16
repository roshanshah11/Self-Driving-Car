import numpy as np
import time

class RacingController:
    def __init__(self, car_params=None):
        # Default car parameters
        self.car_params = car_params or {
            'max_velocity': 5.0,  # m/s
            'max_acceleration': 2.0,  # m/s^2
            'max_deceleration': 4.0,  # m/s^2
            'min_turn_radius': 0.5,  # meters
            'wheelbase': 0.3,  # meters
            'mass': 3.0,  # kg
            'max_steer_angle': 30,  # degrees
        }
        
        # PID controller parameters with improved values
        self.steering_pid = {
            'kp': 1.2,  # Increased proportional gain for more responsive steering
            'ki': 0.05,  # Small integral gain to eliminate steady-state error
            'kd': 0.3,  # Increased derivative gain for better damping
            'prev_error': 0.0,  # Previous error for derivative term
            'integral': 0.0,  # Integral accumulator
            'integral_limit': 5.0,  # Limit for integral term to prevent windup
        }
        
        self.speed_pid = {
            'kp': 0.7,  # Increased for more responsive speed control
            'ki': 0.1,  # Integral gain
            'kd': 0.2,  # Increased derivative gain for smoother acceleration/deceleration
            'prev_error': 0.0,  # Previous error for derivative term
            'integral': 0.0,  # Integral accumulator
            'integral_limit': 1.0,  # Limit for integral term
        }
        
        # Current state
        self.current_velocity = 0.0  # m/s
        self.current_steering = 0.0  # degrees
        
        # Target racing line and speed profile
        self.racing_line = None
        self.speed_profile = None
        self.current_target_idx = 0
        
        # Adaptive lookahead distance based on speed
        self.min_lookahead_distance = 0.5  # meters at low speed
        self.max_lookahead_distance = 2.0  # meters at max speed
        self.lookahead_base = 0.8  # base lookahead distance
        self.lookahead_factor = 0.2  # lookahead factor multiplied by speed
        
        # Track progress
        self.lap_start_time = None
        self.lap_count = 0
        self.last_progress = 0
        
    def set_racing_line(self, racing_line, speed_profile=None):
        """
        Set the target racing line and optional speed profile
        """
        self.racing_line = racing_line
        self.speed_profile = speed_profile
        self.current_target_idx = 0
        self.lap_start_time = time.time()
        self.lap_count = 0
        self.last_progress = 0
    
    def calculate_lookahead_distance(self):
        """
        Calculate adaptive lookahead distance based on current speed
        Higher speeds need longer lookahead distances
        """
        # Normalize speed between 0 and 1
        speed_factor = min(1.0, self.current_velocity / self.car_params['max_velocity'])
        
        # Calculate adaptive lookahead distance
        lookahead = self.lookahead_base + self.lookahead_factor * speed_factor * self.car_params['max_velocity']
        
        # Ensure within bounds
        return np.clip(lookahead, self.min_lookahead_distance, self.max_lookahead_distance)
    
    def find_target_point(self, current_position):
        """
        Find target point on racing line based on adaptive lookahead distance
        Uses a pure pursuit approach with track progress tracking
        """
        if self.racing_line is None or len(self.racing_line) < 2:
            return None, 0
            
        # Convert to numpy array if not already
        current_position = np.array(current_position[:2])  # Only use x,y
        
        # Find the nearest point on racing line
        min_dist = float('inf')
        nearest_idx = 0
        
        # Search within a reasonable range from the last target point
        search_width = min(50, len(self.racing_line) // 4)  # Adaptive search width
        search_start = max(0, self.current_target_idx - search_width // 2)
        search_end = min(len(self.racing_line), search_start + search_width)
        
        for i in range(search_start, search_end):
            dist = np.linalg.norm(self.racing_line[i] - current_position)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # If we're near the end of the search window, expand search to entire track
        if nearest_idx == search_end - 1 or nearest_idx == search_start:
            for i in range(len(self.racing_line)):
                dist = np.linalg.norm(self.racing_line[i] - current_position)
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i
        
        # Calculate adaptive lookahead distance based on current speed
        lookahead_distance = self.calculate_lookahead_distance()
        
        # Look ahead from the nearest point
        accumulated_dist = 0
        lookahead_idx = nearest_idx
        
        for i in range(1, len(self.racing_line)):
            idx = (nearest_idx + i) % len(self.racing_line)
            prev_idx = (idx - 1) % len(self.racing_line)
            
            # Calculate distance between consecutive points
            segment_dist = np.linalg.norm(self.racing_line[idx] - self.racing_line[prev_idx])
            accumulated_dist += segment_dist
            
            if accumulated_dist >= lookahead_distance:
                lookahead_idx = idx
                break
        
        # Track progress and lap counting
        progress_percent = (nearest_idx / len(self.racing_line)) * 100
        
        # Detect lap completion
        if self.last_progress > 80 and progress_percent < 20:
            self.lap_count += 1
            lap_time = time.time() - self.lap_start_time
            print(f"Lap {self.lap_count} completed in {lap_time:.2f} seconds")
            self.lap_start_time = time.time()
        
        self.last_progress = progress_percent
        
        # Update current target index for next iteration
        self.current_target_idx = nearest_idx
        
        # Get target speed at this point with interpolation for smoother transitions
        target_speed = 0
        if self.speed_profile is not None:
            if lookahead_idx < len(self.speed_profile):
                # Consider future speeds for better anticipation
                target_speed = self.speed_profile[lookahead_idx]
                
                # Optionally: Look at the next few points to anticipate changes
                preview_dist = 3  # Number of points to look ahead
                if lookahead_idx + preview_dist < len(self.speed_profile):
                    next_speeds = self.speed_profile[lookahead_idx:lookahead_idx+preview_dist]
                    min_upcoming_speed = min(next_speeds)
                    
                    # If slowing down is needed soon, start slowing down earlier
                    if min_upcoming_speed < target_speed * 0.8:
                        # Blend current target speed with upcoming minimum
                        target_speed = 0.7 * target_speed + 0.3 * min_upcoming_speed
            else:
                target_speed = self.speed_profile[0]  # Use start of profile if index is out of bounds
        else:
            # Default speed if no profile available
            target_speed = self.car_params['max_velocity'] * 0.5  
            
        return self.racing_line[lookahead_idx], target_speed
    
    def calculate_steering_angle(self, current_position, current_heading, target_point):
        """
        Calculate steering angle to reach target point
        Uses pure pursuit algorithm with improved coordinate handling
        """
        if target_point is None:
            return 0.0
            
        # Extract position x,y
        current_x, current_y = current_position[:2]
        target_x, target_y = target_point
        
        # Calculate relative position in car's frame
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Rotate to car's frame
        # Convert heading to radians for trigonometric functions
        heading_rad = current_heading
        cos_heading = np.cos(-heading_rad)
        sin_heading = np.sin(-heading_rad)
        
        # Transform target point to car's coordinate frame
        car_x = dx * cos_heading - dy * sin_heading
        car_y = dx * sin_heading + dy * cos_heading
        
        # Calculate curvature (1/R)
        # For pure pursuit: curvature = 2y/L²
        # where y is lateral offset and L is lookahead distance
        lookahead_dist = np.sqrt(dx*dx + dy*dy)
        if lookahead_dist < 0.1:
            return 0.0  # Avoid division by zero
            
        curvature = 2.0 * car_y / (lookahead_dist * lookahead_dist)
        
        # Convert curvature to steering angle
        # For Ackermann steering: tan(steering) = L * curvature
        # where L is wheelbase
        wheelbase = self.car_params['wheelbase']
        steering_angle = np.arctan(wheelbase * curvature) * 180.0 / np.pi
        
        # Limit steering angle
        max_steer = self.car_params['max_steer_angle']
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)
        
        # Add speed-dependent steering limitation for stability at high speeds
        speed_factor = min(1.0, self.current_velocity / self.car_params['max_velocity'])
        max_rate_of_change = 30.0 * (1.0 - 0.5 * speed_factor)  # degrees per second
        
        # Limit steering change rate based on speed
        max_change = max_rate_of_change * 0.1  # Assuming dt = 0.1s
        prev_steering = self.current_steering
        steering_angle = np.clip(steering_angle, 
                               prev_steering - max_change,
                               prev_steering + max_change)
        
        return steering_angle
    
    def pid_control(self, pid_params, error, dt):
        """
        Enhanced PID control implementation with anti-windup
        """
        # Proportional term
        p_term = pid_params['kp'] * error
        
        # Integral term with anti-windup
        pid_params['integral'] += error * dt
        
        # Apply integral limiting to prevent windup
        if 'integral_limit' in pid_params:
            pid_params['integral'] = np.clip(
                pid_params['integral'], 
                -pid_params['integral_limit'], 
                pid_params['integral_limit']
            )
            
        i_term = pid_params['ki'] * pid_params['integral']
        
        # Derivative term with filtering to reduce noise sensitivity
        d_term = pid_params['kd'] * (error - pid_params['prev_error']) / max(dt, 0.001)
        
        # Update previous error
        pid_params['prev_error'] = error
        
        # Sum all terms
        output = p_term + i_term + d_term
        
        return output
    
    def update_control(self, current_position, current_heading, dt=0.1):
        """
        Update control signals based on current state and target racing line
        Returns (throttle, steering) control signals
        """
        # Find target point on racing line
        target_point, target_speed = self.find_target_point(current_position)
        
        if target_point is None:
            return 0.0, 0.0  # No throttle, no steering
            
        # Calculate steering angle to target
        raw_steering = self.calculate_steering_angle(current_position, current_heading, target_point)
        
        # Apply PID control to steering for smoother control
        steering_error = raw_steering - self.current_steering
        steering_adjustment = self.pid_control(self.steering_pid, steering_error, dt)
        new_steering = self.current_steering + steering_adjustment
        
        # Limit steering angle
        max_steer = self.car_params['max_steer_angle']
        new_steering = np.clip(new_steering, -max_steer, max_steer)
        
        # Speed control with anticipatory logic
        speed_error = target_speed - self.current_velocity
        
        # Adjust throttle response based on steering angle
        # Reduce speed in sharp turns
        steering_factor = 1.0 - (abs(new_steering) / max_steer) * 0.5
        adjusted_target = target_speed * steering_factor
        adjusted_error = adjusted_target - self.current_velocity
        
        # Calculate throttle using PID
        throttle = self.pid_control(self.speed_pid, adjusted_error, dt)
        
        # Limit throttle between -1 and 1
        throttle = np.clip(throttle, -1.0, 1.0)
        
        # Update current state
        self.current_steering = new_steering
        
        # More sophisticated vehicle dynamics model
        if throttle > 0:
            # Accelerating
            accel = throttle * self.car_params['max_acceleration']
        else:
            # Braking
            accel = throttle * self.car_params['max_deceleration']
            
        # Update velocity using acceleration and time step
        self.current_velocity += accel * dt
        
        # Ensure velocity stays within physical limits
        self.current_velocity = np.clip(self.current_velocity, 0, self.car_params['max_velocity'])
        
        return throttle, new_steering