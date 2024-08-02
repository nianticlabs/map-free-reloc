import logging
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from lazy_camera import LazyCamera
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pyrender

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from skimage import io
import shutil

from render_util import get_position_marker, generate_grid, generate_frustum_at_position, get_image_box

_logger = logging.getLogger(__name__)

OVERALL_SCALE_MAPPING = 4.  # size of the mapping image
OVERALL_SCALE_QUERY = 2.  # size of the query frustum


def convert_cv_to_gl(pose):
    """
    Convert a pose from OpenCV coordinate system to OpenGL coordinate system.

    This function takes a 4x4 transformation matrix representing a pose in the OpenCV coordinate system
    and converts it to the OpenGL coordinate system by applying a predefined transformation matrix.

    Args:
        pose (numpy.ndarray): A 4x4 transformation matrix representing the pose in the OpenCV coordinate system.

    Returns:
        numpy.ndarray: A 4x4 transformation matrix representing the pose in the OpenGL coordinate system.
    """

    gl_to_cv = np.array([[1, -1, -1, 1], [-1, 1, 1, -1], [-1, 1, 1, -1], [1, 1, 1, 1]])
    return gl_to_cv * pose


def load_poses(file_path):
    """
    Load poses from a file.

    This function reads a file containing pose data, parses the data, and converts it into a dictionary of poses.
    Each pose is represented as a 4x4 transformation matrix. If a confidence value is provided, it is included in the dictionary.

    Args:
        file_path (str): The path to the file containing the pose data.

    Returns:
        dict: A dictionary where the keys are file names and the values are tuples containing the 4x4 transformation matrix and the confidence value.
    """

    # read pose file
    with open(file_path, "r") as f:
        pose_data = f.readlines()

    poses = {}

    # parse pose file
    for pose_string in pose_data:

        # ignore comments
        if pose_string.startswith("#"):
            continue

        # split pose string into tokens
        pose_string = pose_string.split()

        file_name = pose_string[0]

        # convert quaternion to rotation matrix
        pose_q = pose_string[1:5]
        pose_q = np.array([float(pose_q[1]), float(pose_q[2]), float(pose_q[3]), float(pose_q[0])])

        # check whether the pose is valid. Test poses are all zero
        if np.all(pose_q == 0):
            poses[file_name] = (None, None)
            continue

        pose_R = Rotation.from_quat(pose_q).as_matrix()

        # read translation
        pose_t = np.array(pose_string[5:8])

        # construct full pose matrix
        pose_4x4 = np.identity(4)
        pose_4x4[0:3, 0:3] = pose_R
        pose_4x4[0:3, 3] = pose_t

        # convert world->cam to cam->world
        pose_4x4 = np.linalg.inv(pose_4x4)
        pose_4x4 = convert_cv_to_gl(pose_4x4)

        # check if a confidence is provided
        if len(pose_string) == 9:
            confidence = float(pose_string[-1])
            poses[file_name] = (pose_4x4, confidence)
        else:
            poses[file_name] = (pose_4x4, math.inf)

    return poses


def get_retro_colors():
    # create custom color map
    cdict = {
        'red': [
            [0.0, 0.073, 0.073],
            [0.4, 0.325, 0.325],
            [0.7, 0.286, 0.286],
            [0.85, 0.266, 0.266],
            [0.95, 0, 0],
            [1, 1, 1],
        ],
        'green': [
            [0.0, 0.0, 0.0],
            [0.4, 0.058, 0.058],
            [0.7, 0.470, 0.470],
            [0.85, 0.827, 0.827],
            [0.95, 1, 1],
            [1, 1, 1],
        ],
        'blue': [
            [0.0, 0.057, 0.057],
            [0.4, 0.223, 0.223],
            [0.7, 0.752, 0.752],
            [0.85, 0.988, 0.988],
            [0.95, 1, 1],
            [1, 1, 1],
        ]}

    retroColorMap = LinearSegmentedColormap('retroColors', segmentdata=cdict, N=256)
    return retroColorMap(np.linspace(0, 1, 256))[:, :3]


def render_trajectory(r, trajectory, camera, camera_pose, frustum_images):
    """
    Render a scene with a trajectory, camera, and frustum images.

    This function creates a `pyrender.Scene` object, adds the provided trajectory, camera, and frustum images to the scene,
    and then renders the scene using the provided renderer.

    Args:
        r (pyrender.OffscreenRenderer): The renderer used to render the scene.
        trajectory (pyrender.Node): The trajectory (frustums, markers, etc) to be added to the scene.
        camera (pyrender.Camera): The camera to render from.
        camera_pose (numpy.ndarray): The pose of the camera.
        frustum_images (list): A list of frustum images to be added to the scene.

    Returns:
        numpy.ndarray: The rendered color image.
    """
    scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(1, 1, 1))
    scene.add(trajectory)
    scene.add(camera, pose=camera_pose)

    for frustum_image in frustum_images:
        scene.add(frustum_image)

    color, _ = r.render(scene, flags=(pyrender.constants.RenderFlags.RGBA | pyrender.constants.RenderFlags.FLAT))

    return color


def blend_images(img1_RGB, img2_RGBA, weight=1):
    """
    Blend two images using the alpha channel of the second image.

    This function blends two images by using the alpha channel of the second image (`img2_RGBA`) as a mask.
    The blending is controlled by the `weight` parameter, which scales the alpha mask.

    Args:
        img1_RGB (numpy.ndarray): The first image in RGB format.
        img2_RGBA (numpy.ndarray): The second image in RGBA format, where the alpha channel is used as a mask.
        weight (float, optional): A scaling factor for the alpha mask. Defaults to 1.

    Returns:
        numpy.ndarray: The blended image in RGB format.
    """
    mask = img2_RGBA[:, :, 3].astype(float)
    mask /= 255
    mask = np.expand_dims(mask, axis=2) * weight

    blended_rgb = img2_RGBA[:, :, :3].astype(float) * mask + img1_RGB.astype(float) * (1 - mask)
    return blended_rgb.astype('uint8')


def run_cmd(cmd, raise_on_error=True, verbose=True):
    """
    Executes a command in a subprocess and prints its output to stdout.

    Args:
        cmd (list): The command to be executed, represented as a list of strings.
        raise_on_error (bool, optional): If True, raises a RuntimeError if the command returns a non-zero exit code.
                                          Defaults to True.
        verbose (bool, optional): If True, the output of the subprocess is printed to stdout. Defaults to True.

    Returns:
        int: The return code of the executed command.

    Raises:
        RuntimeError: If the command returns a non-zero exit code and raise_on_error is True.
    """

    # Convert each element of the command to a string
    cmd_str = [str(c) for c in cmd]

    # Start a subprocess with the command
    proc = subprocess.Popen(cmd_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Continuously read and print the output of the subprocess to stdout
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        # If verbose is True, print the output of the subprocess to stdout
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()

    # Wait for the subprocess to finish and get its return code
    returncode = proc.wait()

    # If the return code is non-zero and raise_on_error is True, raise a RuntimeError
    if returncode != 0 and raise_on_error:
        raise RuntimeError("Error running ACE0: \nCommand:\n" + " ".join(cmd_str))

    # Return the return code of the subprocess
    return returncode


def render_scene(pose_file, scene_folder, target_dir, confidence_threshold):
    """
    The function renders a video of the estimates in pose file using the images in scene_folder.
    First, the function renders individual frames and then combines them into a video using ffmpeg.
    The video file will be stored in the execution directory.

    If ground truth poses are available in the data_folder, it will be rendered alongside the estimates.

    @param pose_file: The path to the file containing the estimated poses, in the map-free benchmark format.
    @param scene_folder: The path to the folder containing the images of the map-free dataset scene.
    @param target_dir: The path to the folder where the rendered frames will be stored.
    @param confidence_threshold: Filter estimates below this confidence threshold (position markers not kept)
    """

    scene_id = pose_file.stem[5:]

    # create a temporary directory for the rendered frames
    tmp_target_folder = target_dir / f"tmp_{scene_id}"
    os.makedirs(tmp_target_folder, exist_ok=True)

    # maximum error for color mapping, in meters
    # green to yellow up to max error, red beyond
    error_scale = 1.0

    # number of frames to wait before showing the grid
    grid_wait = 30
    # number of frames to wait before showing the estimates
    est_wait = 90

    # output image resolution
    render_width = 1280
    render_height = 720

    # get reference image and reference pose
    mapping_image_path = f"{scene_folder}/seq0/frame_00000.jpg"
    mapping_pose = np.eye(4)

    # get list of all jpg images in scene folder (query images)
    query_image_paths = sorted(list((scene_folder / "seq1").glob("*.jpg")))
    query_frame_count = len(query_image_paths)

    # load ground truth query poses
    query_pose_path = f"{scene_folder}/poses.txt"
    query_poses_gt = load_poses(query_pose_path)

    _logger.info(f"Found {len(query_poses_gt)} ground truth poses in {query_pose_path}")

    # load estimated poses from pose file
    query_poses_est = load_poses(pose_file)

    _logger.info(f"Found {len(query_poses_est)} estimated poses in {pose_file}")

    # show progress bar while rendering
    p_bar = tqdm(range(query_frame_count + est_wait))

    # setup lazy scene camera (smooth view interpolation)
    scene_camera = LazyCamera()

    # initialise pyrender pipeline
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=render_width / render_height)
    r = pyrender.OffscreenRenderer(render_width, render_height, point_size=4.0)

    # color map for the grid
    retro_cmap = get_retro_colors()

    # color map for estimates (if ground truth is available)
    error_cmap = plt.cm.get_cmap("summer")(np.linspace(0, 1, 256))[:, :3]  # green to yellow
    error_cmap[255] = (0.8, 0, 0)  # red for outliers

    mapping_pose_color = (240, 0, 229)
    query_pose_color = retro_cmap[int(0.95 * 255)] * 255

    # holds position markers for all estimated poses so far
    position_markers = []

    # frustum of the mapping/reference image
    frustum_ref = generate_frustum_at_position(rotation=mapping_pose[:3, :3], translation=mapping_pose[:3, 3],
                                               color=mapping_pose_color, size=OVERALL_SCALE_MAPPING,
                                               border_only=True)
    trajectory_trimesh_ref = pyrender.Mesh.from_trimesh([frustum_ref])

    image_mesh_ref = get_image_box(image_path=mapping_image_path,
                                   frustum_pose=mapping_pose,
                                   flip=True,
                                   cam_marker_size=OVERALL_SCALE_MAPPING)
    image_mesh_ref = pyrender.Mesh.from_trimesh(image_mesh_ref)

    # main rendering loop
    for frame_idx in p_bar:

        # index of estimate (stays 0 for the first few frames)
        est_idx = max(0, frame_idx - est_wait)

        # get image and ground truth pose for the current estimate
        query_image_path = query_image_paths[est_idx]
        # separate image name with sequence folder, and the remaining path
        query_image_parent = query_image_path.parent.parent
        query_image_path = str(Path(query_image_path.parent.name) / query_image_path.name)

        # get GT pose of this image, dismiss confidence value
        query_pose_gt = query_poses_gt.get(query_image_path, (None, None))[0]

        # get estimated pose of this image
        query_pose_est, query_pose_est_conf = query_poses_est.get(query_image_path, (None, None))

        # skip frames without ground truth
        if query_pose_gt is None and query_pose_est is None:
            _logger.warning(f"Neither ground truth pose nor estimate found for {query_image_path}, skipping.")
            continue

        if query_pose_gt is None:
            # there is no ground truth pose but an estimate, e.g. for the map-free test set
            # we treat the estimate as ground truth, and do not render a separate estiamte
            query_pose_gt = query_pose_est
            query_pose_est = None

        # initialise observing camera
        if frame_idx < est_wait:
            # in the beginning, we focus on the mapping frame
            scene_camera.update_camera(mapping_pose.copy())
        else:
            # after the initial frames, we follow the estimated poses
            scene_camera.update_camera(query_pose_gt.copy())

        # frustum of the query ground truth
        frustum_gt = generate_frustum_at_position(rotation=query_pose_gt[:3, :3], translation=query_pose_gt[:3, 3],
                                                  color=query_pose_color, size=OVERALL_SCALE_QUERY)

        # get images as renderable objects of the query and the reference
        image_meshes = []

        image_mesh = get_image_box(image_path=query_image_parent / query_image_path,
                                   frustum_pose=query_pose_gt,
                                   flip=True,
                                   cam_marker_size=OVERALL_SCALE_QUERY)
        image_mesh = pyrender.Mesh.from_trimesh(image_mesh)
        image_meshes.append(image_mesh)
        image_meshes.append(image_mesh_ref)

        # get grid for this frame, grid will be partially transparent in the first few frames
        grid = generate_grid(frame_idx - grid_wait, retro_cmap)

        if query_pose_est is not None:

            # get color of query frustum
            if query_pose_est_conf < confidence_threshold:
                # show estimates below the confidence as grey frustum
                err_color = (100, 100, 100)
            else:
                # calculate error as position error
                err = np.linalg.norm(query_pose_gt[:3, 3] - query_pose_est[:3, 3])
                norm_err = max(min(err / error_scale, 1.0), 0)
                err_color = error_cmap[int(norm_err * 255)] * 255

                # keep position markers starting from the second estimate
                if est_idx > 0:
                    position_markers.append(get_position_marker(query_pose_est, err_color))

            # add estimated query frustum to the objects to draw
            frustum_est = generate_frustum_at_position(rotation=query_pose_est[:3, :3],
                                                       translation=query_pose_est[:3, 3], color=err_color,
                                                       size=OVERALL_SCALE_QUERY)

            trajectory_trimesh = pyrender.Mesh.from_trimesh([frustum_gt, frustum_est] + position_markers[:-1])

        else:
            # draw only the ground truth frustum and the previous position markers
            trajectory_trimesh = pyrender.Mesh.from_trimesh([frustum_gt] + position_markers[:-1])

        # get smooth observing camera
        smooth_camera_pose = scene_camera.get_current_view()

        # we start with a black image
        bg_RGB = np.zeros((render_height, render_width, 3))

        # start by adding the grid (not available in the first few frames)
        if grid is None:
            blended_RGB = bg_RGB
        else:
            # draw grid and paste onto BG
            grid_mesh = pyrender.Mesh.from_trimesh([grid])
            grid_RGBA = render_trajectory(r, grid_mesh, camera, smooth_camera_pose, [])
            blended_RGB = blend_images(bg_RGB, grid_RGBA)

        # render camera reference image
        cams_RGBA = render_trajectory(r, trajectory_trimesh_ref, camera, smooth_camera_pose, image_meshes[1:])
        blended_RGB = blend_images(blended_RGB, cams_RGBA)

        # render query estimates
        cams_RGBA = render_trajectory(r, trajectory_trimesh, camera, smooth_camera_pose, image_meshes[:1])

        # add query estimates in, but with a smooth fade in the beginning of the video
        blend_weight = max(0, min(est_idx/15, 1))
        blended_RGB = blend_images(blended_RGB, cams_RGBA, blend_weight)

        # save result
        out_render_file = f"{tmp_target_folder}/frame_{frame_idx:05d}.png"
        io.imsave(out_render_file, blended_RGB)

        # display progress bar
        p_bar.set_description(f"Rendering {out_render_file}")

    # get ffmpeg path
    ffmpeg_path = shutil.which("ffmpeg")

    # run ffmpeg to convert the rendered images to a video
    run_cmd([ffmpeg_path,
             "-y",
             "-framerate", 30,
             "-pattern_type", "glob",
             "-i", f"{tmp_target_folder}/*.png",
             "-c:v", "libx264",
             "-pix_fmt", "yuv420p",
             target_dir / f"{scene_id}.mp4"
            ])

    # remove temporary directory
    shutil.rmtree(tmp_target_folder)