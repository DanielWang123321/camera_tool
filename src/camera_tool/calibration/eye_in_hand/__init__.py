"""Eye-in-Hand calibration solver, session manager, and waypoint planner."""
from camera_tool.calibration.eye_in_hand.session import CalibrationSession
from camera_tool.calibration.eye_in_hand.solver import (
    OPENCV_HAND_EYE_METHODS,
    evaluate_board_geometry,
    evaluate_depth_consistency,
    evaluate_solution,
    solve_hand_eye_all,
)
from camera_tool.calibration.eye_in_hand.waypoints import WaypointPlanner

__all__ = [
    "CalibrationSession",
    "OPENCV_HAND_EYE_METHODS",
    "WaypointPlanner",
    "evaluate_board_geometry",
    "evaluate_depth_consistency",
    "evaluate_solution",
    "solve_hand_eye_all",
]
