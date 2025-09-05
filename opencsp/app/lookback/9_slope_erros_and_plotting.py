import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from os.path import join, dirname
import gzip
import json
import random as rng
from datetime import datetime, timezone, timedelta
import ast  # For safe parsing of string tuples
from scipy.interpolate import griddata


import imageio.v3 as imageio
import cv2 as cv


import opencsp.common.lib.tool.image_tools as it
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

# Specify the folder where the log file should be saved
log_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/0_checkpoints/error_logs"  # Replace with your desired folder path
log_file = join(log_folder, "error_log_sofast_data_style_create_debug.txt")

# Set up logger
logger = lt.logger(log_dir_body_ext=log_file, level=lt.log.WARN)


'''
from opencsp.app.sofast.lib.DisplayShape import DisplayShape as Display
from opencsp.app.sofast.lib.DefinitionFacet import DefinitionFacet
from opencsp.app.sofast.lib.Fringes import Fringes
from opencsp.app.sofast.lib.ImageCalibrationScaling import ImageCalibrationScaling
from opencsp.app.sofast.lib.MeasurementSofastFringe import MeasurementSofastFringe
from opencsp.app.sofast.lib.ProcessSofastFringe import ProcessSofastFringe as Sofast
from opencsp.app.sofast.lib.SofastConfiguration import SofastConfiguration
from opencsp.app.sofast.lib.SpatialOrientation import SpatialOrientation
from opencsp.common.lib.camera.Camera import Camera
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.deflectometry.Surface2DPlano import Surface2DPlano
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.opencsp_path.opencsp_root_path import opencsp_code_dir

from opencsp.common.lib.cv import SpotAnalysis as sa
from contrib.common.lib.cv.spot_analysis.image_processor.TargetBoardLocatorImageProcessor import (
    TargetBoardLocatorImageProcessor,
)


def lookback_to_sofast_data_format(dark_image, light_image):
    # Define save dir
    dir_save = join(dirname(__file__), "data/output/single_facet")
    ft.create_directories_if_necessary(dir_save)

    # Define sample data directory
    dir_data_sofast = join(opencsp_code_dir(), "test/data/sofast_fringe")
    dir_data_common = join(opencsp_code_dir(), "test/data/sofast_common")

    file_facet = join(dir_data_common, "Facet_NSTTF.json")

    camera = Camera()
    display = Display()
    orientation = SpatialOrientation()
    measurement = MeasurementSofastFringe()
    calibration = ImageCalibrationScaling()
    facet_data = DefinitionFacet.load_from_json(file_facet)

    measurement.mask_images = [dark_image, light_image]
    surface = Surface2DPlano(robust_least_squares=True, downsample=10)

    sofast = Sofast(measurement, orientation, camera, display)

    mirror_refernce = MirrorParametric.generate_flat()

def calculate_homography_transform(output_dir, key_image):
    image_processors = {'Homg': TargetBoardLocatorImageProcessor(key_image, None, 1.2192, 1.2192, 2, 2)}

    image_processors_list = list(image_processors.values())

    spot_analysis = sa.SpotAnalysis("key_image_homography", image_processors_list, save_dir=output_dir)
    spot_analysis.set_primary_images(key_image)
    for result in spot_analysis:
        pass
'''


def find_corners(input_img_color, input_img_gray, max_corners, height_range, width_range):
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.01
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    source_window = 'Image'

    img_copy = np.copy(input_img_color)
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(
        input_img_gray,
        max_corners,
        qualityLevel,
        minDistance,
        None,
        blockSize=blockSize,
        gradientSize=gradientSize,
        useHarrisDetector=useHarrisDetector,
        k=k,
    )

    tl = [width_range[0], height_range[0]]
    tr = [width_range[1], height_range[0]]
    bl = [width_range[0], height_range[1]]
    br = [width_range[1], height_range[1]]
    targets = np.array([tl, tr, bl, br])
    # tlc = find_closest_corners(corners.reshape(-1, 2), np.array(tl))
    # trc = find_closest_corners(corners.reshape(-1, 2), np.array(tr))
    # blc = find_closest_corners(corners.reshape(-1, 2), np.array(bl))
    # brc = find_closest_corners(corners.reshape(-1, 2), np.array(br))
    closest_corners = find_unique_closest_points(corners.reshape(-1, 2), targets)

    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(
            img_copy,
            (int(corners[i, 0, 0]), int(corners[i, 0, 1])),
            radius,
            (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
            cv.FILLED,
        )  # Plotting all detected corners
    for target in targets:
        cv.circle(
            img_copy,
            (int(target[0]), int(target[1])),
            6,
            (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
            2,
        )  # plotting the corners of the rectangular region math was done in
    for found in closest_corners:
        cv.circle(
            img_copy,
            (int(found[0]), int(found[1])),
            8,
            (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
            4,
        )  # plotting the corners closest to the corners of the rectangular region.

    # Create a window
    # cv.namedWindow(source_window)
    # cv.imshow(source_window, img_copy)
    return closest_corners


def find_unique_closest_points(coordinates, target_points):
    """
    Find the closest unique coordinate point to each target point.

    Parameters:
        coordinates (numpy.ndarray): Array of shape (n, 2) representing n pixel coordinates.
        target_points (numpy.ndarray): Array of shape (m, 2) representing m target points.

    Returns:
        numpy.ndarray: Array of shape (m, 2) where each row is the closest unique coordinate point to the corresponding target point.
    """
    closest_points = []
    used_indices = set()  # Keep track of indices that have already been selected

    for target in target_points:
        # Compute the Euclidean distance from the target point to all coordinate points
        distances = np.linalg.norm(coordinates - target, axis=1)

        # Exclude already used indices by setting their distances to infinity
        for idx in used_indices:
            distances[idx] = np.inf

        # Find the index of the closest coordinate point
        closest_index = np.argmin(distances)

        # Append the closest coordinate point to the result list
        closest_points.append(coordinates[closest_index])

        # Mark this index as used
        used_indices.add(closest_index)

    return np.array(closest_points)


def compute_homography(pixel_coords, rect_width, rect_height):
    """
    Compute the homography matrix for perspective correction using real-world dimensions
    and pixel coordinates to minimize distortion.

    Parameters:
        pixel_coords (numpy.ndarray): A 4x2 array of pixel coordinates in the image
                                      corresponding to the corners of the subject.
                                      The order should be top-left, top-right, bottom-right, bottom-left.
        rect_width (float): The width of the subject in real-world dimensions.
        rect_height (float): The height of the subject in real-world dimensions.

    Returns:
        numpy.ndarray: The 3x3 homography matrix.
    """
    if pixel_coords.shape != (4, 2):
        raise ValueError("pixel_coords must be a 4x2 array of corner coordinates.")

    # Compute pixel distances between corners
    top_edge_length = np.linalg.norm(pixel_coords[1] - pixel_coords[0])  # Top-left to top-right
    bottom_edge_length = np.linalg.norm(pixel_coords[2] - pixel_coords[3])  # Bottom-right to bottom-left
    left_edge_length = np.linalg.norm(pixel_coords[3] - pixel_coords[0])  # Bottom-left to top-left
    right_edge_length = np.linalg.norm(pixel_coords[2] - pixel_coords[1])  # Bottom-right to top-right

    # Compute scaling factors based on real-world dimensions
    scale_x_top = rect_width / top_edge_length
    scale_x_bottom = rect_width / bottom_edge_length
    scale_y_left = rect_height / left_edge_length
    scale_y_right = rect_height / right_edge_length

    # Adjust destination points based on real-world dimensions and scaling factors
    rect_coords = np.array(
        [
            [0, 0],  # Top-left corner
            [rect_width * scale_x_top, 0],  # Top-right corner
            [rect_width * scale_x_bottom, rect_height * scale_y_right],  # Bottom-right corner
            [0, rect_height * scale_y_left],  # Bottom-left corner
        ],
        dtype=np.float32,
    )

    # Compute the homography matrix
    homography_matrix, _ = cv.findHomography(pixel_coords.astype(np.float32), rect_coords)

    return homography_matrix


def compute_perspective_transform(pixel_coords):
    # Define the source points (perfectly centered square)
    source_points = np.array(
        [
            [860, 440],  # Top-left corner
            [1060, 440],  # Top-right corner
            [1060, 640],  # Bottom-right corner
            [860, 640],  # Bottom-left corner
        ],
        dtype=np.float32,
    )
    source_points_flip = np.array(
        [[860, 440], [1060, 440], [860, 640], [1060, 640]], dtype=np.float32  # Top-left corner  # Top-right corner
    )

    # Compute the perspective transformation matrix
    perspective_matrix = cv.findHomography(pixel_coords.astype(np.float32), source_points_flip, method=1)

    return perspective_matrix, source_points_flip


def warp_image_with_full_view(image, perspective_matrix, rect_width, rect_height, forced_points, plot_forced_pts=False):
    """
    Warp the image using the perspective transformation matrix and ensure the output image
    is large enough to show the entire transformed image.

    Parameters:
        image (numpy.ndarray): Input image.
        perspective_matrix (numpy.ndarray): 3x3 perspective transformation matrix.
        rect_width (float): The width of the subject in real-world dimensions.
        rect_height (float): The height of the subject in real-world dimensions.

    Returns:
        numpy.ndarray: Warped image.
    """
    # Convert real-world dimensions to pixel dimensions
    output_width = int(rect_width)
    output_height = int(rect_height)

    # Warp the image
    warped_image = cv.warpPerspective(image, perspective_matrix[0], (output_width, output_height))
    if plot_forced_pts:
        for point in forced_points:
            cv.circle(
                warped_image,
                (int(point[0]), int(point[1])),
                8,
                (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
                4,
            )

    # cv.imshow("Warped Image", warped_image)

    return warped_image


def apply_homography_to_pixel_and_vectors(homography_matrix, pixel_coord, normal_vectors):
    """
    Applies a homography matrix to a pixel coordinate and adjusts associated 3D normal vectors.

    Parameters:
        homography_matrix (numpy.ndarray): 3x3 homography matrix.
        pixel_coord (tuple or list): (x, y) pixel coordinate.
        normal_vectors (list of numpy.ndarray): List of 3D normal vectors (nx, ny, nz).

    Returns:
        transformed_coord (numpy.ndarray): Transformed pixel coordinate in 3D (x', y', z').
        transformed_normals (list of numpy.ndarray): List of transformed 3D normal vectors.
    """
    # Ensure homography matrix is a numpy array
    H = np.array(homography_matrix)

    # Step 1: Transform the pixel coordinate
    # Convert pixel coordinate to 3D homogeneous coordinates (add z=1)
    pixel_coord_3d = np.array([pixel_coord[0], pixel_coord[1], 1])

    # Apply the homography matrix
    transformed_coord_homogeneous = H @ pixel_coord_3d

    # Convert back to Cartesian coordinates by dividing by the third (homogeneous) component
    transformed_coord = transformed_coord_homogeneous / transformed_coord_homogeneous[2]

    # Step 2: Adjust normal vectors
    transformed_normals = []

    # Extract the rotational component of the homography matrix
    rotation_matrix = H[:3, :3]  # Extract the 3x3 rotation part

    # Normalize the rotation matrix to remove scaling (if needed)
    scaling_factor = np.linalg.norm(rotation_matrix[:, 0])  # Calculate scaling factor
    rotation_matrix_normalized = rotation_matrix / scaling_factor

    # Apply the rotation matrix to each normal vector
    for normal_vector in normal_vectors:
        # Ensure the normal vector is a numpy array
        normal_vector = np.array(normal_vector)

        # Apply rotation to the 3D normal vector
        rotated_normal = rotation_matrix_normalized @ normal_vector

        # Normalize the transformed normal vector to ensure it remains a unit vector
        rotated_normal /= np.linalg.norm(rotated_normal)

        transformed_normals.append(rotated_normal)

    return transformed_coord, transformed_normals


def warp_mirror_pixels_and_data(
    compressed_json_path, homography_matrix, warped_image, original_image, target_points_for_warp
):
    data = read_compressed_json(compressed_json_path)
    # chkpt_data = read_json(chkpt_data_path)
    warped_points = []
    for point in target_points_for_warp:
        new_pixel_coords, _ = apply_homography_to_pixel_and_vectors(
            homography_matrix=homography_matrix, pixel_coord=point, normal_vectors=np.array([[0, 0, 1], [1, 0, 0]])
        )
        warped_points.append(new_pixel_coords)
    warped_points = np.array(warped_points)

    new_data = dict(data)
    for pixel, details in new_data.items():
        if isinstance(details, dict):
            if "slope_1" in details or "slope_2" in details:
                if details["slope_1"].size < 3 or details["slope_2"].size < 3:
                    # no Calculated Slope
                    new_data[pixel].update({"plot_color": "m"})
                    continue
                elif np.array_equal(details["slope_1"], details["slope_2"]):
                    # Calculated Slope without overlap so two equal slope estimated
                    # with midpoint between circle centers
                    pix_pt = ast.literal_eval(pixel)
                    temp_slopes = np.array([details["slope_1"], details["slope_2"]])
                    new_pixel_coords, new_pixel_normals = apply_homography_to_pixel_and_vectors(
                        homography_matrix=homography_matrix,
                        pixel_coord=[pix_pt[1], pix_pt[0]],  # previously np.array(ast.literal_eval(pixel))
                        normal_vectors=temp_slopes,
                    )
                    slope_differences, slope_errors = calculate_slope_differences(
                        reference_vector=details["observer_vector"],
                        sample_vectors=np.array([details["slope_1"], details["slope_2"]]),
                    )

                    new_data[pixel].update(
                        {
                            "plot_color": "g",
                            "transformed_coords": new_pixel_coords,
                            "transformed_normals": new_pixel_normals,
                            "compared_to": "observer_vector",
                            "slope_differences": slope_differences,
                            "slope_errors": slope_errors,
                        }
                    )
                else:
                    # Calculated Slope with overlap so two unique slopes estimated
                    pix_pt = ast.literal_eval(pixel)
                    temp_slopes = np.array([details["slope_1"], details["slope_2"]])
                    new_pixel_coords, new_pixel_normals = apply_homography_to_pixel_and_vectors(
                        homography_matrix=homography_matrix,
                        pixel_coord=[pix_pt[1], pix_pt[0]],  # previously np.array(ast.literal_eval(pixel))
                        normal_vectors=temp_slopes,
                    )
                    slope_differences, slope_errors = calculate_slope_differences(
                        reference_vector=details["observer_vector"],
                        sample_vectors=np.array([details["slope_1"], details["slope_2"]]),
                    )
                    new_data[pixel].update(
                        {
                            "plot_color": "b",
                            "transformed_coords": new_pixel_coords,
                            "transformed_normals": new_pixel_normals,
                            "compared_to": "observer_vector",
                            "slope_differences": slope_differences,
                            "slope_errors": slope_errors,
                        }
                    )
            else:
                continue

    # Initialize figures and axes
    fig1 = plt.figure(figsize=(10, 8))
    wi = fig1.add_subplot(111)
    wi.imshow(warped_image, cmap="gray")

    fig2 = plt.figure(figsize=(10, 8))
    oi = fig2.add_subplot(111)
    oi.imshow(original_image, cmap="gray")

    # Prepare lists for batch plotting
    original_green = []
    original_blue = []
    original_red = []
    warped_green = []
    warped_blue = []

    progress_bar = tqdm(total=len(new_data), desc="Collecting Data for Comparison for Warped Image")

    for pixel, details in new_data.items():
        # Parse pixel coordinates safely
        row, col = ast.literal_eval(pixel)  # Replace eval with ast.literal_eval

        if isinstance(details, dict):
            if details["slope_1"].size < 3 or details["slope_2"].size < 3:
                original_red.append((col, row))
                continue
            else:
                transformed_coords = details["transformed_coords"]
                if details["plot_color"] == "g":
                    original_green.append((col, row))
                    warped_green.append(transformed_coords)
                elif details["plot_color"] == "b":
                    original_blue.append((col, row))
                    warped_blue.append(transformed_coords)
        else:
            original_red.append((col, row))

        progress_bar.update(1)

    progress_bar.close()

    # Convert lists to NumPy arrays for efficient plotting
    original_green = np.array(original_green)
    original_blue = np.array(original_blue)
    original_red = np.array(original_red)
    warped_green = np.array(warped_green)
    warped_blue = np.array(warped_blue)

    # Batch plot points
    if original_green.size > 0:
        oi.scatter(
            original_green[:, 0],
            original_green[:, 1],
            c="g",
            s=1,
            marker="s",
            alpha=0.5,
            label="Pixels without Overlap",
        )
    if original_green.size > 0:
        oi.scatter(
            original_blue[:, 0], original_blue[:, 1], c="b", s=1, marker="s", alpha=0.5, label="Pixels With Overlap"
        )
    if original_red.size > 0:
        oi.scatter(original_red[:, 0], original_red[:, 1], c="r", s=1, marker="s", alpha=0.1, label="All Pixels")
    if warped_green.size > 0:
        wi.scatter(
            warped_green[:, 0], warped_green[:, 1], c="g", s=1, marker="s", alpha=0.5, label="Pixels without Overlap"
        )
    if warped_blue.size > 0:
        wi.scatter(warped_blue[:, 0], warped_blue[:, 1], c="b", s=1, marker="s", alpha=0.5, label="Pixels With Overlap")

    wi.scatter(
        warped_points[:, 0],
        warped_points[:, 1],
        c="purple",
        s=15,
        marker="o",
        alpha=0.5,
        label="Warped Homography Points",
    )

    wi.legend()
    oi.legend()
    # plt.show()
    return new_data


def visualize_transform_matrix_changes(matrix):
    # 1. Define the transformation matrix
    M = np.array(matrix)

    # Perform SVD
    U, s, VT = np.linalg.svd(M)
    Sigma = np.diag(s)

    # 2. Visualize scaling
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Create a unit sphere
    phi, theta = np.mgrid[0 : np.pi : 20j, 0 : 2 * np.pi : 20j]
    x_sphere = np.sin(phi) * np.cos(theta)
    y_sphere = np.sin(phi) * np.sin(theta)
    z_sphere = np.cos(phi)

    # Plot original sphere
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='b', alpha=0.3)
    ax1.set_title('Original Unit Sphere')
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])

    # Plot scaled ellipsoid
    x_scaled = s[0] * x_sphere
    y_scaled = s[1] * y_sphere
    z_scaled = s[2] * z_sphere
    ax2.plot_surface(x_scaled, y_scaled, z_scaled, color='r', alpha=0.6)
    ax2.set_title('Scaled Ellipsoid (after SVD scaling)')
    # ax2.set_xlim([-2, 2])
    # ax2.set_ylim([-2, 2])
    # ax2.set_zlim([-2, 2])
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    plt.show()

    # 3. Visualize rotation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Original axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='x_orig')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='y_orig')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='z_orig')

    # Transformed axes
    M_rot = U @ VT
    x_prime = M_rot @ np.array([1, 0, 0])
    y_prime = M_rot @ np.array([0, 1, 0])
    z_prime = M_rot @ np.array([0, 0, 1])

    ax.quiver(0, 0, 0, x_prime[0], x_prime[1], x_prime[2], color='r', linestyle='--', label='x_rot')
    ax.quiver(0, 0, 0, y_prime[0], y_prime[1], y_prime[2], color='g', linestyle='--', label='y_rot')
    ax.quiver(0, 0, 0, z_prime[0], z_prime[1], z_prime[2], color='b', linestyle='--', label='z_rot')

    ax.set_title('Rotation of Coordinate Axes')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


def calculate_slope_differences(reference_vector, sample_vectors):
    """
    Compare a reference vector with multiple sample vectors and return the differences and errors in all 3 dimensions.

    Parameters:
        reference_vector (list or np.ndarray): A 3D vector [x, y, z] representing the reference.
        sample_vectors (list or np.ndarray): A list or array of 3D vectors [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
        dict: A dictionary containing the differences and errors for each sample vector.
              {
                  "differences": [[dx1, dy1, dz1], [dx2, dy2, dz2], ...],
                  "errors": [error1, error2, ...]
              }
    """
    # Ensure inputs are numpy arrays for easier manipulation
    reference_vector = np.array(reference_vector)
    sample_vectors = np.array(sample_vectors)

    # Validate dimensions
    if reference_vector.shape != (3,):
        raise ValueError("Reference vector must be a 3D vector [x, y, z].")
    if sample_vectors.ndim != 2 or sample_vectors.shape[1] != 3:
        raise ValueError("Sample vectors must be a list or array of 3D vectors [[x, y, z], ...].")

    # Calculate differences
    differences = sample_vectors - reference_vector

    # Calculate errors (Euclidean distance)
    errors_mag = np.linalg.norm(differences, axis=1)

    return differences.tolist(), errors_mag.tolist()


def plot_error_contours(data_dict):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_differences": List of up to two sets of [dx, dy, dz].
                    - "slope_errors": List of up to two magnitudes of error.
                - Empty lists for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set1, y_coords_set1, slope_errors_set1 = [], [], []
    x_errors_set1, y_errors_set1, z_errors_set1 = [], [], []

    x_coords_set2, y_coords_set2, slope_errors_set2 = [], [], []
    x_errors_set2, y_errors_set2, z_errors_set2 = [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            # Extract the first set of data
            if data["slope_1"].size < 3 or data["slope_2"].size < 3:
                continue
            elif len(data["slope_errors"]) > 0:
                x_coords_set1.append(data["transformed_coords"][0])
                y_coords_set1.append(data["transformed_coords"][1])
                slope_errors_set1.append(data["slope_errors"][0])
                x_errors_set1.append(data["slope_differences"][0][0])
                y_errors_set1.append(data["slope_differences"][0][1])
                z_errors_set1.append(data["slope_differences"][0][2])

            # Extract the second set of data (if available)
            if len(data["slope_errors"]) > 1:
                x_coords_set2.append(data["transformed_coords"][0])
                y_coords_set2.append(data["transformed_coords"][1])
                slope_errors_set2.append(data["slope_errors"][1])
                x_errors_set2.append(data["slope_differences"][1][0])
                y_errors_set2.append(data["slope_differences"][1][1])
                z_errors_set2.append(data["slope_differences"][1][2])
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")
    '''
    scale_max = np.max(
        [
            x_errors_set1,
            y_errors_set1,
            z_errors_set1,
            slope_errors_set1,
            x_errors_set2,
            y_errors_set2,
            z_errors_set2,
            slope_errors_set2,
        ]
    )
    scale_min = np.min(
        [
            x_errors_set1,
            y_errors_set1,
            z_errors_set1,
            slope_errors_set1,
            x_errors_set2,
            y_errors_set2,
            z_errors_set2,
            slope_errors_set2,
        ]
    )
    '''

    # Helper function to plot a set of data
    def plot_set(x_coords, y_coords, slope_errors, x_errors, y_errors, z_errors, set_label):
        # Convert lists to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        slope_errors = np.array(slope_errors)
        x_errors = np.array(x_errors)
        y_errors = np.array(y_errors)
        z_errors = np.array(z_errors)

        # Create a grid for contour plotting
        grid_x, grid_y = np.meshgrid(
            np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
        )

        # Interpolate errors onto the grid
        error_grid = {
            "Magnitude": griddata((x_coords, y_coords), slope_errors, (grid_x, grid_y), method="linear"),
            "X Error": griddata((x_coords, y_coords), x_errors, (grid_x, grid_y), method="linear"),
            "Y Error": griddata((x_coords, y_coords), y_errors, (grid_x, grid_y), method="linear"),
            "Z Error": griddata((x_coords, y_coords), z_errors, (grid_x, grid_y), method="linear"),
        }

        # Plot each error type as a contour plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        error_types = ["Magnitude", "X Error", "Y Error", "Z Error"]
        for ax, error_type in zip(axes.flat, error_types):
            contour = ax.contourf(
                grid_x, grid_y, error_grid[error_type], cmap="jet", levels=1000  # vmin=scale_min, vmax=scale_max
            )
            cbar = plt.colorbar(contour, ax=ax)
            ax.set_title(f"{error_type} Contour Plot ({set_label})")
            ax.set_xlabel("X Pixel Location")
            ax.set_ylabel("Y Pixel Location")
            cbar.set_label(f"{error_type}")
            ax.invert_yaxis()

        plt.tight_layout()
        # plt.show()

    # Plot the first set of data
    if x_coords_set1:
        plot_set(x_coords_set1, y_coords_set1, slope_errors_set1, x_errors_set1, y_errors_set1, z_errors_set1, "Set 1")

    # Plot the second set of data
    if x_coords_set2:
        plot_set(x_coords_set2, y_coords_set2, slope_errors_set2, x_errors_set2, y_errors_set2, z_errors_set2, "Set 2")


def save_checkpoint(checkpoint_folder, checkpoint_file_name, checkpoint_data, print_path=True):
    """
    Saves the checkpoint data to a JSON file.

    Parameters:
        checkpoint_file_name (str): Path to the checkpoint file.
        checkpoint_data (dict): Dictionary containing checkpoint information.

    Returns:
        None
    """
    raw_path = os.path.join(checkpoint_folder, checkpoint_file_name)
    norm_path = os.path.normpath(raw_path)

    success = False
    for i in range(10):
        try:
            with open(norm_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=4)
                success = True
                break
        except Exception:
            logger.error("Unable to save checkpoint", exc_info=True)
            time.sleep(0.01)

    if success:
        if print_path and i == 0:
            logger.info("Checkpoint saved to %s", raw_path)
        if i > 0:
            logger.info("Checkpoint saved to %s after %d attempts", raw_path, i)


def load_checkpoint(checkpoint_folder, checkpoint_file_name):
    """
    Loads the checkpoint data from a JSON file.

    Parameters:
        checkpoint_folder (str): Path to the folder containing the checkpoint file.
        checkpoint_file_name (str): Name of the checkpoint file.

    Returns:
        dict: Dictionary containing checkpoint information, or None if the file is empty or does not exist.
    """
    norm_path = os.path.normpath(os.path.join(checkpoint_folder, checkpoint_file_name))

    if os.path.exists(norm_path):
        # Check if the file is empty
        if os.path.getsize(norm_path) == 0:
            print(f"Checkpoint file {norm_path} is empty.")
            return None

        # Attempt to load the JSON data
        with open(norm_path, 'r', encoding='utf-8') as f:
            try:
                checkpoint_data = json.load(f)
                # Check if the loaded JSON is empty (e.g., {} or [])
                if not checkpoint_data:
                    print(f"Checkpoint file {norm_path} contains empty JSON data.")
                    return None
                print(f"Checkpoint loaded from {norm_path}")
                return checkpoint_data
            except json.JSONDecodeError:
                logger.error("Invalid Json File", exc_info=True)
                print(f"Checkpoint file {norm_path} contains invalid JSON.")
                return None
    else:
        print(f"Checkpoint file {norm_path} does not exist.")
        return None


def custom_serializer(obj):
    """
    Custom serializer to handle numpy arrays, datetime objects, floats, integers, and strings.
    Converts unsupported types into JSON-compatible formats.
    """
    try:
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list

        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()  # Convert datetime to ISO 8601 string

        # Handle floats
        elif isinstance(obj, float):
            return float(obj)  # Ensure it's a float (JSON-compatible)

        # Handle integers
        elif isinstance(obj, int):
            return int(obj)  # Ensure it's an integer (JSON-compatible)

        # Handle strings
        elif isinstance(obj, str):
            return str(obj)  # Ensure it's a string (JSON-compatible)

        # Handle sets (convert to list for JSON compatibility)
        elif isinstance(obj, set):
            return list(obj)  # Convert set to list

        # Unsupported type
        else:
            logger.error("Unsupported type encountered: %r", type(obj), exc_info=True)
            raise TypeError(f"Type {type(obj)} not serializable")

    except Exception:
        logger.error("Error during serialization", exc_info=True)
        raise


def custom_deserializer(obj):
    """
    Custom deserializer to handle strings, datetime strings, numpy arrays, floats, and integers.
    Converts JSON-compatible formats back into their original types.
    """
    for key, value in obj.items():
        # Handle numpy arrays (lists of numbers)
        if isinstance(value, list) and all(isinstance(i, (int, float)) for i in value):
            try:
                obj[key] = np.array(value)  # Convert list back to numpy array
            except ValueError:
                logger.error("Failed to convert %s to numpy array", key, exc_info=True)
                # If conversion fails, leave as list

        # Handle datetime strings
        elif isinstance(value, str):
            try:
                obj[key] = datetime.fromisoformat(value)  # Convert ISO 8601 string back to datetime
            except ValueError:
                logger.debug("%s is not a valid datetime string, leaving as string.", key, exc_info=True)
                # If conversion fails, leave as string

        # Handle floats explicitly
        elif isinstance(value, float):
            obj[key] = float(value)  # Ensure it's a float (redundant but explicit)

        # Handle integers explicitly
        elif isinstance(value, int):
            obj[key] = int(value)  # Ensure it's an integer (redundant but explicit)

        # Handle strings explicitly (fallback case)
        elif isinstance(value, str):
            obj[key] = str(value)  # Ensure it's a string (redundant but explicit)

    return obj


def read_json(file_path):
    """
    Reads a regular JSON file and returns the data.

    Parameters:
        file_path (str): Path to the .json file.

    Returns:
        dict or list: Data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'r', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_json Error", exc_info=True)
        return None


def write_json(data, file_path):
    """
    Writes data to a regular JSON file.

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    try:
        with open(norm_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, default=custom_serializer)
        print(f"Data written to {norm_path}")
    except Exception:
        logger.error("write_json Error", exc_info=True)


def read_compressed_json(file_path):
    """
    Reads a compressed JSON file (.json.gz) and returns the data.

    Parameters:
        file_path (str): Path to the .json.gz file.

    Returns:
        list: List of dictionaries containing the data from the file.
    """
    norm_path = os.path.normpath(file_path)
    try:
        with gzip.open(norm_path, 'rt', encoding='utf-8') as f:
            data = json.load(f, object_hook=custom_deserializer)
        return data
    except Exception:
        logger.error("read_compressed_json Error", exc_info=True)
        return None


def write_compressed_json(data, file_path, print_path=True):
    """
    Writes data to a compressed JSON file (.json.gz).

    Parameters:
        data (dict or list): Data to write to the file.
        file_path (str): Path to the .json.gz file.

    Returns:
        None
    """
    norm_path = os.path.normpath(file_path)
    success = False
    for i in range(10):
        try:
            with gzip.open(norm_path, 'wt', encoding='utf-8') as f:
                try:
                    json.dump(data, f, indent=4, default=custom_serializer)
                    success = True
                    break
                except Exception:
                    logger.error("write_compressed_json Error", exc_info=True)
                    time.sleep(0.01)
        except Exception:
            logger.error("write_compressed_json Error", exc_info=True)

    if success:
        if print_path and i == 0:
            logger.info("File Saved to %s", norm_path)
        if i > 0:
            logger.info("File saved to %s after %d attempts", norm_path, i)


if __name__ == "__main__":
    json_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    output_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/9_sofast_data_compare"
    dark_mask_image = "//snl//Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/1_video_frames/DSC_0025-00001.png"
    light_mask_image = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025/3_specific_cropped_frames\DSC_0025-23490.png"
    dark_mask = cv.imread(dark_mask_image)
    light_mask = cv.imread(light_mask_image)
    light_mask_gray = cv.cvtColor(light_mask, cv.COLOR_BGR2GRAY)

    height_range = [435, 670]
    width_range = [830, 1070]

    facet_corners = find_corners(
        input_img_color=light_mask,
        input_img_gray=light_mask_gray,
        max_corners=10,
        height_range=height_range,
        width_range=width_range,
    )

    # flipped_corners = np.array(
    #     [
    #         [facet_corners[0, 1], facet_corners[0, 0]],
    #         [facet_corners[1, 1], facet_corners[1, 0]],
    #         [facet_corners[2, 1], facet_corners[2, 0]],
    #         [facet_corners[3, 1], facet_corners[3, 0]],
    #     ]
    # )

    transform_matrix, forced_transform_points = compute_perspective_transform(facet_corners)
    warped_image = warp_image_with_full_view(
        image=light_mask,
        perspective_matrix=transform_matrix,
        rect_width=light_mask.shape[1],
        rect_height=light_mask.shape[0],
        forced_points=forced_transform_points,
    )

    visualize_transform_matrix_changes(transform_matrix[0])

    new_data = warp_mirror_pixels_and_data(
        compressed_json_path=json_data_location,
        homography_matrix=transform_matrix[0],
        warped_image=warped_image,
        original_image=light_mask,
        target_points_for_warp=facet_corners,
    )

    plot_error_contours(new_data)
    plt.show()
