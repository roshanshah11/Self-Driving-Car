import json
import time

class GeneralResult:
    def __init__(self, results):
        self.barcode = results.get("Barcode", [])
        self.classifierResults = [ClassifierResult(item) for item in results.get("Classifier", [])]
        self.detectorResults = [DetectorResult(item) for item in results.get("Detector", [])]
        self.fiducialResults = [FiducialResult(item) for item in results.get("Fiducial", [])]
        self.retroResults = [RetroreflectiveResult(item) for item in results.get("Retro", [])]
        self.botpose = results.get("botpose", [])
        self.botpose_wpiblue = results.get("botpose_wpiblue", [])
        self.botpose_wpired = results.get("botpose_wpired", [])
        self.capture_latency = results.get("cl", 0)
        self.pipeline_id = results.get("pID", 0)
        self.robot_pose_target_space = results.get("t6c_rs", [])
        self.targeting_latency = results.get("tl", 0)
        self.timestamp = results.get("ts", 0)
        self.validity = results.get("v", 0)
        self.parse_latency = 0.0


class RetroreflectiveResult:
    def __init__(self, retro_data):
        self.points = retro_data.get("pts", [])
        self.camera_pose_target_space = retro_data.get("t6c_ts", [])
        self.robot_pose_field_space = retro_data.get("t6r_fs", [])
        self.robot_pose_target_space = retro_data.get("t6r_ts", [])
        self.target_pose_camera_space = retro_data.get("t6t_cs", [])
        self.target_pose_robot_space = retro_data.get("t6t_rs", [])
        self.target_area = retro_data.get("ta", 0)
        self.target_x_degrees = retro_data.get("tx", 0)
        self.target_x_pixels = retro_data.get("txp", 0)
        self.target_y_degrees = retro_data.get("ty", 0)
        self.target_y_pixels = retro_data.get("typ", 0)


class FiducialResult:
    def __init__(self, fiducial_data):
        self.fiducial_id = fiducial_data.get("fID", 0)
        self.family = fiducial_data.get("fam", "")
        self.points = fiducial_data.get("pts", [])
        self.skew = fiducial_data.get("skew", 0)
        self.camera_pose_target_space = fiducial_data.get("t6c_ts", [])
        self.robot_pose_field_space = fiducial_data.get("t6r_fs", [])
        self.robot_pose_target_space = fiducial_data.get("t6r_ts", [])
        self.target_pose_camera_space = fiducial_data.get("t6t_cs", [])
        self.target_pose_robot_space = fiducial_data.get("t6t_rs", [])
        self.target_area = fiducial_data.get("ta", 0)
        self.target_x_degrees = fiducial_data.get("tx", 0)
        self.target_x_pixels = fiducial_data.get("txp", 0)
        self.target_y_degrees = fiducial_data.get("ty", 0)
        self.target_y_pixels = fiducial_data.get("typ", 0)


class DetectorResult:
    def __init__(self, detector_data):
        self.class_name = detector_data.get("class", "")
        self.class_id = detector_data.get("classID", 0)
        self.confidence = detector_data.get("conf", 0)
        self.points = detector_data.get("pts", [])
        self.target_area = detector_data.get("ta", 0)
        self.target_x_degrees = detector_data.get("tx", 0)
        self.target_x_pixels = detector_data.get("txp", 0)
        self.target_y_degrees = detector_data.get("ty", 0)
        self.target_y_pixels = detector_data.get("typ", 0)


class ClassifierResult:
    def __init__(self, classifier_data):
        self.class_name = classifier_data.get("class", "")
        self.class_id = classifier_data.get("classID", 0)
        self.confidence = classifier_data.get("conf", 0)


def parse_results(json_data):
    """
    Parse Limelight results JSON data into structured classes
    """
    start_time = time.time()
    if json_data is not None:
        parsed_result = GeneralResult(json_data)
        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        parsed_result.parse_latency = elapsed_time_ms
        return parsed_result
    return None


def extract_white_lines(parsed_result):
    """
    Extract white line contours from Limelight results
    Specifically designed for track boundary detection
    """
    if parsed_result is None:
        return None, None
    
    # Check for retroreflective results (white lines)
    if parsed_result.retroResults and len(parsed_result.retroResults) >= 2:
        # Sort by area (largest first)
        retro_results = sorted(parsed_result.retroResults, key=lambda x: x.target_area, reverse=True)
        
        # Get the two largest contours (assuming they are inner and outer boundaries)
        outer_boundary = retro_results[0].points if retro_results[0].points else None
        inner_boundary = retro_results[1].points if len(retro_results) > 1 and retro_results[1].points else None
        
        return inner_boundary, outer_boundary
    
    # If no retroreflective results, check detector results as fallback
    elif parsed_result.detectorResults and len(parsed_result.detectorResults) >= 2:
        # Sort by area (largest first)
        detector_results = sorted(parsed_result.detectorResults, key=lambda x: x.target_area, reverse=True)
        
        # Get the two largest contours
        outer_boundary = detector_results[0].points if detector_results[0].points else None
        inner_boundary = detector_results[1].points if len(detector_results) > 1 and detector_results[1].points else None
        
        return inner_boundary, outer_boundary
    
    return None, None