import os
import time
import numpy as np
from logging import DEBUG, ERROR
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
import ast


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt

from opencsp.common.lib.camera.Camera import Camera

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render.view_spec as vs
import opencsp.app.sofast.lib.spatial_processing as sp
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz


# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookfast_camera_adjust.txt",
    log_type=DEBUG,
)


def rotation_matrix_scipy(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2 using SciPy.
    """
    # SciPy works with 1D arrays for single vectors
    a = vec1.reshape(-1)
    b = vec2.reshape(-1)

    # The align_vectors method returns a Rotation object and a rmsd value
    rotation_object, rssd = Rotation.align_vectors(b, a)  # Note: order may need adjustment based on specific use case

    # Convert the Rotation object to a 3x3 matrix
    # rotation_matrix = rotation.as_matrix()

    return rotation_object, rssd


def rotate_vector(vector, axis, degrees):
    """
    Rotates a vector about a specified axis by a given number of degrees.

    Parameters:
        vector (numpy.ndarray): The vector to rotate (1D array of shape (3,)).
        axis (numpy.ndarray): The axis to rotate about (1D array of shape (3,)).
        degrees (float): The angle in degrees to rotate the vector.

    Returns:
        numpy.ndarray: The rotated vector (1D array of shape (3,)).
    """
    try:
        # Validate inputs
        if not isinstance(vector, np.ndarray) or vector.shape != (3,):
            logger.error("Invalid vector: Must be a numpy array of shape (3,).")
            raise ValueError("Vector must be a numpy array of shape (3,).")

        if not isinstance(axis, np.ndarray) or axis.shape != (3,):
            logger.error("Invalid axis: Must be a numpy array of shape (3,).")
            raise ValueError("Axis must be a numpy array of shape (3,).")

        if not isinstance(degrees, (int, float)):
            logger.error("Invalid degrees: Must be a numeric value.")
            raise ValueError("Degrees must be a numeric value.")

        # Normalize the axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0:
            logger.error("Invalid axis: Cannot be a zero vector.")
            raise ValueError("Axis cannot be a zero vector.")
        axis = axis / axis_norm

        # Convert degrees to radians
        radians = np.deg2rad(degrees)

        # Compute rotation matrix using Rodrigues' rotation formula
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)
        one_minus_cos = 1 - cos_theta

        # Outer product of axis with itself
        axis_outer = np.outer(axis, axis)

        # Cross-product matrix of axis
        axis_cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])

        # Rotation matrix
        rotation_matrix = cos_theta * np.eye(3) + one_minus_cos * axis_outer + sin_theta * axis_cross

        # Rotate the vector
        rotated_vector = np.dot(rotation_matrix, vector)
        return rotated_vector

    except Exception as e:
        logger.exception("An error occurred while rotating the vector.")
        raise e


def create_rotation_object(axis, angle_degrees):
    """
    Create a scipy Rotation object for a rotation about a specified axis.

    Parameters:
        axis (array-like): A 3D vector specifying the axis of rotation (e.g., [1, 0, 0]).
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        scipy.spatial.transform.Rotation: A Rotation object representing the rotation.
    """
    # Normalize the axis to ensure it is a unit vector
    axis = np.array(axis)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be the zero vector.")
    axis_normalized = axis / np.linalg.norm(axis)

    # Create the rotation object using the axis-angle representation
    rotation = Rotation.from_rotvec(np.radians(angle_degrees) * axis_normalized)

    return rotation


def binary_search_angle(vector, axis, function, tolerance=1e-6, max_iterations=1000):
    """
    Performs a binary search to find the angle that minimizes the z-component of the rotated vector.

    Parameters:
        vector (numpy.ndarray): The vector to rotate (1D array of shape (3,)).
        axis (numpy.ndarray): The axis to rotate about (1D array of shape (3,)).
        rotate_vector (function): Function to rotate the vector.
        tolerance (float): The tolerance for the z-component to be considered zero.
        max_iterations (int): Maximum number of iterations for the binary search.

    Returns:
        float: The angle in degrees that minimizes the z-component of the rotated vector.
    """
    # Define the search range for the angle (0 to 360 degrees)
    low = 0.0
    high = 360.0

    for iteration in range(max_iterations):
        # Calculate the midpoint angle
        mid = (low + high) / 2.0

        # Rotate the vector at the midpoint angle
        rotated_vector = function(vector, axis, mid)

        # Check the z-component of the rotated vector
        z_component = rotated_vector[2]

        if abs(z_component) < tolerance:
            # If the z-component is close to zero, return the angle
            return mid

        # Update the search range based on the sign of the z-component
        if z_component > 0:
            high = mid
        else:
            low = mid

    # If the search did not converge, return the midpoint of the final range
    return (low + high) / 2.0


def ransac_average_direction(vectors, num_iterations=100, tolerance=0.00025):
    """
    Compute a RANSAC-style average direction from a set of 3D vectors.

    Parameters:
        vectors (list or np.ndarray): Array of shape (N, 3) containing [x, y, z] direction components.
        num_iterations (int): Number of RANSAC iterations to perform.
        tolerance (float): Angular tolerance (in radians) for consensus evaluation.

    Returns:
        np.ndarray: The "average" direction vector with the highest consensus.
    """
    # Ensure input is a NumPy array
    vectors = np.array(vectors)

    # Normalize all vectors to unit length
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms

    best_consensus_count = 0
    best_average_direction = None

    for _ in range(num_iterations):
        # Randomly sample a subset of vectors
        sample_indices = np.random.choice(
            len(normalized_vectors), size=round(len(normalized_vectors) / 20), replace=False
        )
        sample_vectors = normalized_vectors[sample_indices]

        # Compute the average direction of the sample
        average_direction = np.mean(sample_vectors, axis=0)
        average_direction /= np.linalg.norm(average_direction)  # Normalize to unit length

        # Compute angular distance between average direction and all vectors
        dot_products = np.dot(normalized_vectors, average_direction)
        angular_distances = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip to avoid numerical issues

        # Count how many vectors are within the tolerance
        consensus_count = np.sum(angular_distances < tolerance)

        # Update the best result if this iteration has higher consensus
        if consensus_count > best_consensus_count:
            best_consensus_count = consensus_count
            best_average_direction = average_direction

    return best_average_direction


def calculate_angle_between_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Calculates the angle (in radians) between two unit vectors in 3D space.

    Parameters:
        vector1 (np.ndarray): A 3D unit vector [x, y, z].
        vector2 (np.ndarray): A 3D unit vector [x, y, z].

    Returns:
        float: The angle between the two vectors in radians.
    """
    # Ensure the input vectors are unit vectors
    if not np.isclose(np.linalg.norm(vector1), 1.0):
        raise ValueError("vector1 is not a unit vector.")
    if not np.isclose(np.linalg.norm(vector2), 1.0):
        raise ValueError("vector2 is not a unit vector.")

    # Compute the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)

    # Clamp the dot product to the range [-1, 1] to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Calculate the angle using arccos
    angle = np.arccos(dot_product)

    return angle


def calculate_slope(observer_vec_start, observer_vec_end, inter_1, inter_2):
    observer_vec = (observer_vec_start + observer_vec_end) / 2
    slope_1 = (observer_vec + inter_1) / np.linalg.norm(observer_vec + inter_1)
    slope_2 = (observer_vec + inter_2) / np.linalg.norm(observer_vec + inter_2)
    return observer_vec, slope_1, slope_2


def calculate_slope_differences(reference_vector, sample_vectors):
    """
    Compare a reference vector with multiple sample vectors and return the differences and errors in all 3 dimensions.

    Parameters:
        reference_vector (list or np.ndarray): A 3D vector [x, y, z] representing the reference.
        sample_vectors (list or np.ndarray): A list or array of 3D vectors [[x1, y1, z1], [x2, y2, z2], ...].

    Returns:
        lists: A list containing the differences and errors for each sample vector.
                  "differences": [[dx1, dy1, dz1], [dx2, dy2, dz2], ...],
                  "errors": [error1, error2, ...]
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


def safe_calculate_slope(observer_vector_start, observer_vector_end, intersection_1, intersection_2):
    """
    Wrapper for calculate_slope with error handling and logging.
    """
    try:
        # Call the calculate_slope function
        observer_vec, slope_1, slope_2 = calculate_slope(
            observer_vector_start, observer_vector_end, intersection_1, intersection_2
        )
        return observer_vec, slope_1, slope_2
    except KeyError as e:
        # Handle missing keys in dictionaries
        logger.debug("KeyError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except TypeError as e:
        # Handle type-related issues (e.g., NoneType or invalid types)
        logger.debug("TypeError in calculate_slope: %s", e, exc_info=True)
        return None, None, None
    except Exception as e:
        # Catch any other unexpected errors
        logger.debug("Unexpected error in calculate_slope: %s", e, exc_info=True)
        return None, None, None


def get_pixel_pointing_vector(pixel_directions, row, col, imagewidth):
    # pixel_pointing[row * light_image.shape[1] + col
    temp = pixel_directions[int(row * imagewidth + col)]
    return temp.data


def plot_slope_heat_maps(data_dict):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            # Extract the first set of data
            if data["slope_1"].size < 3 or data["slope_2"].size < 3:
                continue
            elif len(data["slope_1"]) > 0:
                row, col = ast.literal_eval(pixel)
                x_coords_set.append(col)
                y_coords_set.append(row)
                slope_set1.append(data["slope_1"])
                slope_set2.append(data["slope_2"])
                angle_between.append(data["angle_between"])
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(x_coords_set, y_coords_set, angle_between)

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set1, "Set 1")

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set2, "Set 2")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Radians")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Radians")


def plot_slope_heat_maps_updated(data_dict):
    """
    Iterates through a dictionary with pixel coordinates as keys, extracts up to two sets of data,
    and plots contour plots for each set separately.

    Parameters:
        pixel_dict (dict): A dictionary where:
            - Keys are pixel coordinates (tuples) like (x, y).
            - Values are either:
                - Sub-dictionaries containing:
                    - "slope_1" or "slope_2": List of up to two sets of [nx, ny, nz].
                    - "angle_between": Float of angle between start and end vector.
                - Empty ndarray for pixels without data.

    Returns:
        None: Displays the contour plots for both sets of data.
    """
    # Initialize lists to store data for the first and second sets
    x_coords_set, y_coords_set, slope_set1, slope_set2, angle_between = [], [], [], [], []

    # Iterate through the dictionary
    for pixel, data in data_dict.items():
        if isinstance(data, dict):  # Check if the value is a sub-dictionary
            if data["intersection_1"].size > 0:
                # Extract the first set of data
                if data["slope_1_camera_corrected"].size < 3 or data["slope_2_camera_corrected"].size < 3:
                    continue
                elif len(data["slope_1_camera_corrected"]) > 0:
                    row, col = ast.literal_eval(pixel)
                    x_coords_set.append(col)
                    y_coords_set.append(row)
                    slope_set1.append(data["slope_1_camera_corrected"])
                    slope_set2.append(data["slope_2_camera_corrected"])
                    angle_between.append(data["angle_between"])
            else:
                continue
        elif isinstance(data, np.ndarray) and len(data) == 0:  # Skip empty lists
            continue
        else:
            raise ValueError(f"Unexpected data format for pixel {pixel}: {data}")

    best_slope_1 = ransac_average_direction(np.array(slope_set1))
    best_slope_2 = ransac_average_direction(np.array(slope_set2))

    plot_angle_between_vectors(x_coords_set, y_coords_set, angle_between)

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set1, "Set 1 Updated")

    plot_heat_maps_no_comparison(x_coords_set, y_coords_set, slope_set2, "Set 2 Updated")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Updated")

    plot_heat_maps(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Updated")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set1, best_slope_1, "Set 1 Updated Radians")

    plot_heat_maps_radians(x_coords_set, y_coords_set, slope_set2, best_slope_2, "Set 2 Updated Radians")


def plot_angle_between_vectors(x_coords, y_coords, angles_between):
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    angles = griddata((x_coords, y_coords), angles_between, (grid_x, grid_y), method="linear")

    fig, ax = plt.subplots()
    contour = ax.contourf(grid_x, grid_y, angles, cmap="jet", levels=250)  # vmin=scale_min, vmax=scale_max
    cbar = plt.colorbar(contour, ax=ax)
    ax.set_xlabel("X Pixel Location")
    ax.set_ylabel("Y Pixel Location")
    ax.set_title("Coverage Map Type Plot Heat Map")
    cbar.set_label("Angle Between Vectors [Radians]")
    ax.invert_yaxis()


def plot_heat_maps_no_comparison(x_coords, y_coords, slopes, set_label):
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    # scale_max = np.max([slope_diff_norms])
    # scale_min = np.min([slope_diff_norms])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "X Value": griddata((x_coords, y_coords), slopes[:, 0], (grid_x, grid_y), method="linear"),
        "Y Value": griddata((x_coords, y_coords), slopes[:, 1], (grid_x, grid_y), method="linear"),
        "Z Value": griddata((x_coords, y_coords), slopes[:, 2], (grid_x, grid_y), method="linear"),
    }

    # Plot each error type as a contour plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    error_types = ["X Value", "Y Value", "Z Value"]
    for ax, error_type in zip(axes.flat, error_types):
        contour = ax.contourf(
            grid_x, grid_y, error_grid[error_type], cmap="jet", levels=500  # , vmin=scale_min, vmax=scale_max
        )
        ax.scatter(x_coords, y_coords, s=0.1, c='k', marker='.')  # Actual locations where we have data
        cbar = plt.colorbar(contour, ax=ax)
        ax.set_title(f"{error_type} Heat Map ({set_label})")
        ax.set_xlabel("X Pixel Location")
        ax.set_ylabel("Y Pixel Location")
        cbar.set_label(f"{error_type}")
        ax.invert_yaxis()

    plt.tight_layout()
    # plt.show()


def plot_heat_maps(x_coords, y_coords, slopes, best_slope, set_label):
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    slope_differences, slope_diff_norms = calculate_slope_differences(best_slope, slopes)
    slope_differences = np.array(slope_differences)
    slope_diff_norms = np.array(slope_diff_norms)

    # scale_max = np.max([slope_diff_norms])
    # scale_min = np.min([slope_diff_norms])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "Norm Difference": griddata((x_coords, y_coords), slope_diff_norms, (grid_x, grid_y), method="linear"),
        "X Difference": griddata((x_coords, y_coords), slope_differences[:, 0], (grid_x, grid_y), method="linear"),
        "Y Difference": griddata((x_coords, y_coords), slope_differences[:, 1], (grid_x, grid_y), method="linear"),
        "Z Difference": griddata((x_coords, y_coords), slope_differences[:, 2], (grid_x, grid_y), method="linear"),
    }

    # Plot each error type as a contour plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    error_types = ["Norm Difference", "X Difference", "Y Difference", "Z Difference"]
    for ax, error_type in zip(axes.flat, error_types):
        contour = ax.contourf(
            grid_x, grid_y, error_grid[error_type], cmap="jet", levels=500  # , vmin=scale_min, vmax=scale_max
        )
        cbar = plt.colorbar(contour, ax=ax)
        ax.set_title(f"{error_type} Heat Map ({set_label})")
        ax.set_xlabel("X Pixel Location")
        ax.set_ylabel("Y Pixel Location")
        cbar.set_label(f"{error_type}")
        ax.invert_yaxis()

    plt.tight_layout()
    # plt.show()


def plot_heat_maps_radians(x_coords, y_coords, slopes, best_slope, set_label):
    # Convert lists to numpy arrays
    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    slopes = np.array(slopes)

    slope_deviation_radians = []
    for slope in slopes:
        slope_deviation_radians.append(calculate_angle_between_vectors(best_slope, slope))

    slope_deviation_radians = np.array(slope_deviation_radians)

    # scale_max = np.max([slope_deviation_radians])
    # scale_min = np.min([slope_deviation_radians])
    # Create a grid for contour plotting
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_coords.min(), x_coords.max(), 500), np.linspace(y_coords.min(), y_coords.max(), 500)
    )

    # Interpolate errors onto the grid
    error_grid = {
        "Radian Difference": griddata((x_coords, y_coords), slope_deviation_radians, (grid_x, grid_y), method="linear")
    }

    # Plot each error type as a contour plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    contour = ax.contourf(
        grid_x, grid_y, error_grid["Radian Difference"], cmap="jet", levels=500  # , vmin=scale_min, vmax=scale_max
    )
    cbar = plt.colorbar(contour, ax=ax)
    ax.set_title(f"Difference Heat Map ({set_label})")
    ax.set_xlabel("X Pixel Location")
    ax.set_ylabel("Y Pixel Location")
    cbar.set_label("Radian Difference")
    ax.invert_yaxis()

    plt.tight_layout()
    # plt.show()


def image_to_mirror_coords(pnp_rotation, pnp_translation, vector_data_updated):
    new_pixel_locations = []
    new_slope_1_mirror = []
    new_slope_2_mirror = []
    transform_cMo = 
    for pixel, details in vector_data_updated.items():
        row, col = ast.literal_eval(pixel)
        if isinstance(details, dict):
            if details["intersection_1"].size > 0:
                slope_1 = vector_data_updated[pixel]["slope_1_camera_corrected"]
                slope_2 = vector_data_updated[pixel]["slope_2_camera_corrected"]

                # new_pix_loc = pnp_rotation.apply(np.array([col, row, 0])) + pnp_translation.data.reshape(3)
                new_pix_loc = pnp_rotation.apply(np.array([col, row, 0]) - pnp_translation.data.reshape(3))
                new_slope_1 = pnp_rotation.apply(np.array(slope_1))
                new_slope_2 = pnp_rotation.apply(np.array(slope_2))
            else:
                # new_pix_loc = pnp_rotation.apply(np.array([col, row, 0])) + pnp_translation.data.reshape(3)
                new_pix_loc = pnp_rotation.apply(np.array([col, row, 0]) - pnp_translation.data.reshape(3))
                new_slope_1 = np.array([None, None, None])
                new_slope_2 = np.array([None, None, None])
        else:
            # new_pix_loc = pnp_rotation.apply(np.array([col, row, 0])) + pnp_translation.data.reshape(3)
            new_pix_loc = pnp_rotation.apply(np.array([col, row, 0]) - pnp_translation.data.reshape(3))
            new_slope_1 = np.array([None, None, None])
            new_slope_2 = np.array([None, None, None])

        new_pixel_locations.append(new_pix_loc)
        new_slope_1_mirror.append(new_slope_1)
        new_slope_2_mirror.append(new_slope_2)

    return np.array(new_pixel_locations), np.array(new_slope_1_mirror), np.array(new_slope_2_mirror)


def main():
    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

    plotting = True

    original_data_location = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025_Final_ExEx/8_pixel_vector_information/debug/pixel_vector_information_wslope_debug.json.gz"
    vector_data = lbt.read_compressed_json(original_data_location)

    video_metadata = lbt.extract_detailed_video_metadata(os.path.join(primary_folder, video_name))

    light_image = cv2.imread(os.path.join(primary_folder, "light_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    dark_image = cv2.imread(os.path.join(primary_folder, "dark_mask_test.png"), cv2.IMREAD_GRAYSCALE)
    all_pixels = np.ones(shape=light_image.shape, dtype=bool)

    mask_raw = imgp.calc_mask_raw(
        np.concatenate((dark_image[:, :, np.newaxis], light_image[:, :, np.newaxis]), axis=2),
        hist_thresh=0.5,
        filt_width=9,
        filt_thresh=4,
        thresh_active_pixels=0.01,
    )
    mask = imgp.keep_largest_mask_area(mask_raw)
    v_mask_centroid_image = imgp.centroid_mask(mask)
    v_edges_image = imgp.edges_from_mask(mask)

    mask_image = mask.astype(np.uint8) * 255

    expected_corners_manual = Vxy(
        list(zip([860, 444], [840, 643], [1047, 657], [1062, 455])), dtype=int
    )  # Counterclockwise Starting from Bottom Left Corner in [row, column]

    '''
    
    expected_corners_facet_coords_manual = Vxyz(
        list(zip([-0.606, -0.606, 0], [-0.606, 0.606, 0], [0.606, 0.606, 0], [0.606, -0.606, 0])), dtype=float
    )  # Counterclockwise Starting from Bottom Left Corner in [row, column]
    '''
    expected_corners_facet_coords_manual = Vxyz(
        list(zip([0.606, -0.606, 0], [-0.606, -0.606, 0], [-0.606, 0.606, 0], [0.606, 0.606, 0])), dtype=float
    )  # Counterclockwise Starting from Bottom Right Corner in [row, column] SOFAST Example uses this convention

    v_corners_image = imgp.refine_facet_corners(
        Puv_facet_corns_exp=expected_corners_manual,
        Puv_cent=v_mask_centroid_image,
        Puv_edges=v_edges_image,
        step=20,
        d_perp=20,
        frac_keep=1,
    )

    '''
    # Arbitrary Camera Intrinsic matrix
    K_intrin = np.array([[1, 0, 1920 / 2], [0, 1, 1080 / 2], [0, 0, 1]])
    # Distortion coefficients
    D_coeff = np.array([1, 1, 1, 1])
    '''

    # Sofast Example Camera Intrinsic matrix
    K_intrin = np.array([[5492.064314084441, 0, 1920 / 2], [0, 5486.2706013814895, 1080 / 2], [0, 0, 1]])
    # Sofast Example Camera Distortion coefficients
    D_coeff = np.array([-0.144160742602367, 1.609744377391114, 2.503498158416561e-5, -0.001899042260179])

    cam = Camera(
        intrinsic_mat=K_intrin, distortion_coef=D_coeff, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example"
    )

    r_optic_cam_refine_1, v_cam_optic_cam_refine_1 = sp.calc_rt_from_img_pts(
        pts_image=v_corners_image.vertices, pts_object=expected_corners_facet_coords_manual, camera=cam
    )

    pixel_pointing = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam)

    central_pixel_vector = pixel_pointing[
        int((light_image.shape[0] / 2) * light_image.shape[1]) + int(light_image.shape[1] / 2)
    ]
    # pixel_pointing[row * light_image.shape[1] + col
    pyramid_pixel_vectors = [
        pixel_pointing[int(0 * light_image.shape[1] + 0)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int(0 * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] / 2) * light_image.shape[1] + light_image.shape[1]) - 1],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + 0)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1] / 2)],
        pixel_pointing[int((light_image.shape[0] - 1) * light_image.shape[1] + light_image.shape[1]) - 1],
    ]
    cam_x_axis = Uxyz(np.array([1, 0, 0]))
    cam_y_axis = Uxyz(np.array([0, 1, 0]))

    reference_vector_horizon = Uxyz(vector_data["(500, 900)"]["observer_vector"] * -1)

    rot_obj_no_roll, rssd = rotation_matrix_scipy(
        np.array([central_pixel_vector.x[0], central_pixel_vector.y[0], central_pixel_vector.z[0]]),
        np.array([reference_vector_horizon.x[0], reference_vector_horizon.y[0], reference_vector_horizon.z[0]]),
    )

    cam_xyz_t = rot_obj_no_roll.apply(
        np.array([cam_x_axis.data, cam_y_axis.data, central_pixel_vector.data]).reshape(3, 3)
    )

    test_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    test_output_x = []
    test_output_y = []
    test_output_z = []
    for item in test_vectors:
        test_output_x.append(rotate_vector(item, np.array([1, 0, 0]), degrees=90))
        test_output_y.append(rotate_vector(item, np.array([0, 1, 0]), degrees=90))
        test_output_z.append(rotate_vector(item, np.array([0, 0, 1]), degrees=90))
        test_output_x.append(rotate_vector(item, np.array([1, 0, 0]), degrees=45))
        test_output_y.append(rotate_vector(item, np.array([0, 1, 0]), degrees=45))
        test_output_z.append(rotate_vector(item, np.array([0, 0, 1]), degrees=45))
    '''
    with np.nditer(test_vectors, flags=["external_loop"], op_flags=['readonly']) as it:
        for item in it:
            test_output_x.append(rotate_vector(item, np.array([1, 0, 0]), degrees=90))
            test_output_y.append(rotate_vector(item, np.array([0, 1, 0]), degrees=90))
            test_output_y.append(rotate_vector(item, np.array([0, 0, 1]), degrees=90))
    '''

    roll_control_angle = binary_search_angle(
        vector=cam_xyz_t[0], axis=cam_xyz_t[2], function=rotate_vector, tolerance=1e-6, max_iterations=1000
    )
    roll_control_angle = (180 - roll_control_angle) * -1

    roll_control_obj = create_rotation_object(axis=cam_xyz_t[2], angle_degrees=roll_control_angle)

    cam_xyz_tr = roll_control_obj.apply(cam_xyz_t)

    cam_horizon_transform = roll_control_obj * rot_obj_no_roll

    cam_mirror_transform = r_optic_cam_refine_1 * roll_control_obj * rot_obj_no_roll

    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        vectors_data = [
            [0, 0, 0, cam_x_axis.x[0], cam_x_axis.y[0], cam_x_axis.z[0], "Cam X", "r"],
            [0, 0, 0, cam_y_axis.x[0], cam_y_axis.y[0], cam_y_axis.z[0], "Cam Y", "g"],
            [
                0,
                0,
                0,
                central_pixel_vector.x[0],
                central_pixel_vector.y[0],
                central_pixel_vector.z[0],
                "Cam Z (Central Pixel Pointing)",
                "b",
            ],
            [
                0,
                0,
                0,
                reference_vector_horizon.x[0],
                reference_vector_horizon.y[0],
                reference_vector_horizon.z[0],
                "Horizon Reference",
                "darkorange",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[0][0] * 0.75,
                cam_xyz_t[0][1] * 0.75,
                cam_xyz_t[0][2] * 0.75,
                "Cam X Transformed",
                "maroon",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[1][0] * 0.75,
                cam_xyz_t[1][1] * 0.75,
                cam_xyz_t[1][2] * 0.75,
                "Cam Y Transformed",
                "lime",
            ],
            [
                0,
                0,
                0,
                cam_xyz_t[2][0] * 0.75,
                cam_xyz_t[2][1] * 0.75,
                cam_xyz_t[2][2] * 0.75,
                "Cam Z Transformed",
                "midnightblue",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[0][0] * 0.5,
                cam_xyz_tr[0][1] * 0.5,
                cam_xyz_tr[0][2] * 0.5,
                "Cam X Transformed Roll",
                "aqua",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[1][0] * 0.5,
                cam_xyz_tr[1][1] * 0.5,
                cam_xyz_tr[1][2] * 0.5,
                "Cam Y Transformed Roll",
                "blueviolet",
            ],
            [
                0,
                0,
                0,
                cam_xyz_tr[2][0] * 0.5,
                cam_xyz_tr[2][1] * 0.5,
                cam_xyz_tr[2][2] * 0.5,
                "Cam Z Transformed Roll",
                "deeppink",
            ],
        ]

        for ox, oy, oz, dx, dy, dz, label, color in vectors_data:
            # Plot the vector using quiver
            ax.quiver(ox, oy, oz, dx, dy, dz, color=color, arrow_length_ratio=0.1, label=label)

        # fig_pyr = plt.figure()
        # ax_pyr = fig_pyr.add_subplot(111, projection='3d')
        pix_pyr_t = []
        for index, vec in enumerate(pyramid_pixel_vectors):
            if index == 0:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.15,
                    vec.y * 1.15,
                    vec.z * 1.15,
                    arrow_length_ratio=0.1,
                    color="olivedrab",
                    label="Untransformed Camera Vec Sample",
                )
                pix_pyr_t.append(cam_horizon_transform.apply(vec.data.reshape(3)))
                ax.quiver(
                    0,
                    0,
                    0,
                    pix_pyr_t[index][0] * 1.15,
                    pix_pyr_t[index][1] * 1.15,
                    pix_pyr_t[index][2] * 1.15,
                    arrow_length_ratio=0.1,
                    color="black",
                    label="Transformed Camera Vec Sample",
                )
            else:
                ax.quiver(
                    0,
                    0,
                    0,
                    vec.x * 1.15,
                    vec.y * 1.15,
                    vec.z * 1.15,
                    arrow_length_ratio=0.1,
                    color="olivedrab",
                    label=None,
                )
                pix_pyr_t.append(cam_horizon_transform.apply(vec.data.reshape(3)))
                ax.quiver(
                    0,
                    0,
                    0,
                    pix_pyr_t[index][0] * 1.15,
                    pix_pyr_t[index][1] * 1.15,
                    pix_pyr_t[index][2] * 1.15,
                    arrow_length_ratio=0.1,
                    color="black",
                    label=None,
                )

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim([-1.25, 1.25])
        ax.set_ylim([-1.25, 1.25])
        ax.set_zlim([-1.25, 1.25])
        ax.set_aspect('equal')
        ax.legend()

        for pixel, details in vector_data.items():
            if isinstance(details, dict):
                if details["intersection_1"].size > 0:
                    row, col = ast.literal_eval(pixel)
                    cam_vec_original = get_pixel_pointing_vector(
                        pixel_directions=pixel_pointing, row=row, col=col, imagewidth=mask.shape[1]
                    )
                    cam_vec_horizon = cam_horizon_transform.apply(cam_vec_original.reshape(3))
                    _, slope_1_corr, slope_2_corr = safe_calculate_slope(
                        cam_vec_horizon.reshape(3),
                        cam_vec_horizon.reshape(3),
                        details["intersection_1"],
                        details["intersection_2"],
                    )
                    angle_between = calculate_angle_between_vectors(
                        details['start_vector']['celestial_to_target'], details['end_vector']['celestial_to_target']
                    )
                    vector_data[pixel]["angle_between"] = angle_between
                    vector_data[pixel]["observer_vector_camera_corrected"] = cam_vec_horizon
                    vector_data[pixel]["slope_1_camera_corrected"] = slope_1_corr
                    vector_data[pixel]["slope_2_camera_corrected"] = slope_2_corr
                else:
                    continue
            else:
                pass

        mirror_pixel_coords, mirror_slope_1, mirror_slope_2 = image_to_mirror_coords(
            r_optic_cam_refine_1.inv(), v_cam_optic_cam_refine_1, vector_data
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(mirror_pixel_coords[..., 0], mirror_pixel_coords[..., 1], mirror_pixel_coords[..., 2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plot_slope_heat_maps(data_dict=vector_data)
        plot_slope_heat_maps_updated(data_dict=vector_data)
        plt.show()
        print("done")


if __name__ == "__main__":
    main()
