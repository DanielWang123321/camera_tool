"""Compare camera geometry to the installed RealSense SDK without using hardware."""
import numpy as np
import pytest
import pyrealsense2 as rs

from camera_tool.core import CameraModel, marker_object_points, pose_error, pose_from_mm_deg, transform_points


def camera_metadata(model):
    return {'width': 640, 'height': 480, 'model': model,
            'K': [[394.,0.,320.],[0.,393.,233.],[0.,0.,1.]],
            'distortion': [-0.05355,0.05782,0.0001664,0.0002816,-0.01927]}


def sdk_intrinsics(metadata):
    intr = rs.intrinsics()
    intr.width, intr.height = metadata['width'], metadata['height']
    intr.fx, intr.fy, intr.ppx, intr.ppy = 394.,393.,320.,233.
    intr.coeffs = metadata['distortion']
    intr.model = getattr(rs.distortion, metadata['model'])
    return intr


@pytest.mark.parametrize('model', ['brown_conrady', 'inverse_brown_conrady', 'modified_brown_conrady'])
def test_projection_and_rays_match_realsense(model):
    metadata = camera_metadata(model)
    camera, intr = CameraModel(metadata), sdk_intrinsics(metadata)
    points = np.array([[0.12,0.08,0.3],[-0.12,-0.08,0.3],[0.,0.,0.3],[0.18,-0.1,0.3]])
    pixels = camera.project(points)
    expected = [rs.rs2_project_point_to_pixel(intr, p.tolist()) for p in points]
    assert np.allclose(pixels, expected, atol=6e-5, rtol=0)
    assert np.allclose(camera.deproject_pixels(pixels, points[:,2]), points, atol=2e-7, rtol=0)
    if model != 'modified_brown_conrady':
        expected_points = [rs.rs2_deproject_pixel_to_point(intr, uv.tolist(), float(depth)) for uv, depth in zip(pixels, points[:,2])]
        assert np.allclose(camera.deproject_pixels(pixels, points[:,2]), expected_points, atol=2e-7, rtol=0)


def test_inverse_brown_pnp_recovers_pose_from_distorted_corners():
    camera = CameraModel(camera_metadata('inverse_brown_conrady'))
    pose = pose_from_mm_deg([15,-20,330,175,5,20])
    pixels = camera.project(transform_points(pose, marker_object_points(0.15)))
    recovered, rms = camera.pnp_candidates(pixels, 0.15)[0]
    mm, degrees = pose_error(recovered, pose)
    assert rms < 1e-5 and mm < 1e-4 and degrees < 1e-4
