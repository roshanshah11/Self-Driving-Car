# WhiteLine Track Detection and Mapping System

A robotics vision system for detecting and mapping white track boundaries using a Limelight camera. This system enables robots to build a map of a track marked by white boundary lines and generate an optimal racing path with speed profiles.

## Features

- **Track Detection**: Detects white lines using computer vision techniques
- **Track Mapping**: Builds a 2D map of the track as the robot moves
- **Racing Line Optimization**: Generates optimal racing lines with speed profiles
- **Web Visualization**: Real-time visualization of camera feed and track map
- **Simulation Mode**: Built-in simulation for testing without hardware
- **Configuration System**: Easily customize parameters through a config file
- **Unit and Integration Tests**: Comprehensive test suite

## System Components

- **TrackMapper**: Main class that coordinates the system
- **TrackDetector**: Processes images to detect track boundaries
- **RacingLineOptimizer**: Calculates optimal racing lines with speed profiles
- **AdaptiveRacingLine**: Provides real-time racing line adaptation
- **WebVisualizer**: Provides a web interface for visualization
- **Limelight Integration**: Connects to Limelight camera for vision processing

## Recent Improvements

The codebase has been significantly improved with the following enhancements:

1. **Robust Error Handling**:
   - Added comprehensive exception handling throughout the codebase
   - Implemented graceful degradation when components fail

2. **Logging System**:
   - Added structured logging with file and console output
   - Implemented different log levels (debug, info, warning, error)

3. **Configuration System**:
   - Created centralized configuration with defaults
   - Added support for loading/saving configuration from JSON files

4. **Racing Line Optimization**:
   - Implemented PSO (Particle Swarm Optimization) algorithm for racing line generation
   - Added dynamic speed profile calculation based on vehicle constraints
   - Color-coded visualization of speed profiles for intuitive understanding
   - Support for regenerating racing lines on updated track maps

5. **Testing Infrastructure**:
   - Added unit tests for core components
   - Added integration tests for system functionality
   - Created a test runner with reporting

6. **Code Structure**:
   - Improved method organization
   - Better parameter management
   - Reduced code duplication

## Setup and Installation

### Prerequisites

- Python 3.6+
- OpenCV
- NumPy
- Flask
- NetworkTables
- Limelight camera (or run in simulation mode)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/whiteline-tracking.git
   cd whiteline-tracking
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Running the System

Run in simulation mode (no hardware required):
```bash
python main.py
```

With Limelight hardware:
```bash
python main.py --use-real-camera
```

The web interface will be available at http://localhost:8080

## Configuration

You can customize system parameters by editing the `config.json` file, which will be created automatically on first run. The configuration includes:

- Camera settings (exposure, brightness, threshold values)
- Map settings (size, cell size, starting position)
- Network settings (ports, timeouts)
- Simulation parameters

## Testing

Run the tests using the test runner:
```bash
python run_tests.py
```

Or run individual test files:
```bash
python -m unittest test_track_detector.py
python -m unittest test_track_mapper.py
```

## Future Improvements

Potential areas for future enhancement:

1. **Enhanced Line Detection**:
   - Implement more robust detection algorithms
   - Add adaptive thresholding based on lighting conditions

2. **Racing Line Optimization**:
   - Improve the racing line calculation
   - Add speed profiles for optimal racing

3. **Visualization Enhancements**:
   - Add 3D visualization
   - Implement telemetry dashboards

4. **Map Management**:
   - Save and load maps
   - Track evolution monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.