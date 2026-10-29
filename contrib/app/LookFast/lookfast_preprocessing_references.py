import cv2 as cv
import numpy as np

from contrib.common.lib.cv.RegionDetector import RegionDetector
import opencsp.common.lib.tool.file_tools as ft

import opencsp.common.lib.geometry.LineXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import opencsp.common.lib.geometry.RegionXY as reg2

import contrib.app.LookFast.lookback_tools as lbt


def interactive_region_detection(
    input_image_path: str = None,
    output_dir: str = None,
    RD_canny_edges_gradient: int = 100,
    RD_canny_non_edges_gradient: int = 50,
    RD_edge_coarse_width: int = 50,
    output_ppt_gen: bool = False,
    debug: bool = False,
):
    """
    Perform interactive region detection on an input image.

    Args:
        input_image_path: Path to the input image file.
        output_dir: Directory to save output files.
        RD_canny_edges_gradient: Gradient threshold for edge detection (higher value means fewer edges).
        RD_canny_non_edges_gradient: Gradient threshold for non-edge detection.
        RD_edge_coarse_width: Width for edge coarse detection.
        output_ppt_gen: Whether to generate a PowerPoint output.
        debug: Whether to enable debug mode.

    Returns:
        Dictionary containing details of all detected regions.

    Raises:
        ValueError: If input_image_path is not provided or doesn't exist.
        FileNotFoundError: If the input image file cannot be read.
        Exception: For other unexpected errors during processing.
    """
    # Input validation
    if not input_image_path:
        raise ValueError("input_image_path must be provided")

    if not ft.file_exists(input_image_path):
        raise FileNotFoundError(f"Input image file not found: {input_image_path}")

    if output_dir and not ft.directory_exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    try:
        detector = RegionDetector(
            edge_coarse_width=RD_edge_coarse_width,
            canny_edges_gradient=RD_canny_edges_gradient,
            canny_non_edges_gradient=RD_canny_non_edges_gradient,
            generate_powerpoint=output_ppt_gen,
        )

        # Read the image first to catch any potential errors
        image = cv.imread(input_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image file: {input_image_path}")

        all_region_details = detector.find_multiple_regions_from_interactive_points(
            image=image,
            debug=debug,
            output_powerpoint=ft.join(output_dir, "interactive_region_detection.pptx") if output_ppt_gen else None,
            output_data_dir=output_dir,
        )
        return all_region_details

    except Exception as e:
        raise Exception(f"Error during region detection: {str(e)}")


def load_multiple_regions_from_files(
    regions_json_file: str, masks_json_file: str = None, debug: bool = False
) -> dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]:
    """
    Load previously saved region data from JSON files and optionally a PowerPoint presentation.

    This function loads region data that was saved by the find_multiple_regions_from_interactive_points
    method, including both the region definitions and any masks that were created.

    Parameters
    ----------
    regions_json_file : str
        Path to the regions JSON file created by export_regions_to_json.
    masks_json_file : str, optional
        Path to the masks JSON file created by export_masks_to_json.
    debug : bool, optional
        If True, shows debug information about loaded regions.

    Returns
    -------
    dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]
        A dictionary where keys are region names and values are tuples containing:
        - Dictionary of edge lines for the region
        - Dictionary of corner points
        - The resulting RegionXY object

    Raises
    ------
    FileNotFoundError
        If the regions JSON file doesn't exist.
    ValueError
        If the JSON file has an invalid format.
    """
    detector = RegionDetector(edge_coarse_width=50)

    # Validate input files
    if not ft.file_exists(regions_json_file):
        raise FileNotFoundError(f"Regions JSON file not found: {regions_json_file}")

    if masks_json_file and not ft.file_exists(masks_json_file):
        raise FileNotFoundError(f"Masks JSON file not found: {masks_json_file}")

    # Load regions from JSON
    regions = detector.import_regions_from_json(regions_json_file)

    # Load masks if provided
    masks = []
    if masks_json_file:
        masks = detector.import_masks_from_json(masks_json_file)
        if debug:
            print(f"Loaded {len(masks)} masks from {masks_json_file}")

    if debug:
        print(f"Loaded {len(regions)} regions from {regions_json_file}")
        for region_name in regions:
            print(f"  - {region_name}")

    return regions


def load_and_visualize_regions(
    self, image: np.ndarray, regions_json_file: str, masks_json_file: str = None, debug: bool = False
) -> dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]:
    """
    Load previously saved region data and visualize it on the original image.

    This function loads region data and displays it on the original image for verification.

    Parameters
    ----------
    image : np.ndarray
        The original image to display regions on.
    regions_json_file : str
        Path to the regions JSON file.
    masks_json_file : str, optional
        Path to the masks JSON file.
    debug : bool, optional
        If True, shows debug information and intermediate images.

    Returns
    -------
    dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]
        The loaded regions data.
    """
    # Input validation
    if not image:
        raise ValueError("input_image_path must be provided")

    detector = RegionDetector()
    # Load the regions
    regions = load_multiple_regions_from_files(regions_json_file, masks_json_file, debug)

    # Load masks if provided
    masks = []
    if masks_json_file:
        masks = detector.import_masks_from_json(masks_json_file)

    # Create a display image
    display_image = image.copy()
    if len(display_image.shape) == 2 or display_image.shape[2] == 1:
        display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)

    # Apply masks if any
    if masks:
        mask_image = np.ones_like(image, dtype=np.uint8) * 255  # Start with all white (no mask)
        for mask in masks:
            mask_array = np.array(mask, np.int32)
            cv.fillPoly(mask_image, [mask_array], (50, 50, 50))  # Dark gray for masked areas

        if len(mask_image.shape) == 3:
            mask_image = cv.cvtColor(mask_image, cv.COLOR_BGR2GRAY)

        # Apply mask to display image
        display_image = cv.bitwise_and(display_image, display_image, mask=mask_image)

    # Visualize all regions
    for i, (region_name, (edges, corners, region)) in enumerate(regions.items()):
        # Use the polygon visualization method
        result_image = detector.visualize_polygon_edges_corners(display_image, edges, corners, i, thickness=2)

        # Add region name
        if corners:
            first_corner = next(iter(corners.values()))
            text_pos = (int(first_corner.x[0]) + 10, int(first_corner.y[0]) - 10)
            cv.putText(
                result_image, region_name, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA
            )

        # Show the result
        if debug:
            cv.imshow(f"Region: {region_name}", result_image)
            cv.waitKey(0)
            cv.destroyWindow(f"Region: {region_name}")

    # Show final image with all regions
    if debug:
        cv.imshow("All Loaded Regions", display_image)
        cv.waitKey(0)
        cv.destroyWindow("All Loaded Regions")

    return regions


# Example usage
if __name__ == "__main__":
    # Example parameters - adjust these for your specific use case
    input_image = None

    data_video_path_for_image_scaling = None

    # r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-09-01_HeliostatTestLOOKFAST\3_Post\2026_09_01_C3T01\reference_images\DSC_0003_scaled_to_video_new.JPG"
    output_dir = None

    # r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-09-01_HeliostatTestLOOKFAST\3_Post\2026_09_01_C3T01\reference_images"
    output_ppt = True

    # r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-09-01_HeliostatTestLOOKFAST\3_Post\2026_09_01_C3T01\reference_images\debug_region_detection.pptx"  # Set to None to skip PowerPoint

    canny_edge_gradient_value = 100
    canny_non_edge_gradient_value = 50
    edge_coarse_width_value = 50
    debug_region_detection = False

    if input_image is None:
        input_image = lbt.select_file(
            window_title="Select a LOOKFAST Reference photo taken before or after data collection."
        )

    if data_video_path_for_image_scaling is None:
        data_video_path_for_image_scaling = lbt.select_file(
            window_title="Select a LOOKFAST data video that corresponds to reference image for image scaling."
        )

    if output_dir is None:
        output_dir = lbt.select_dir(
            window_title="Select an output directory for region preprocessing data and debugging info"
        )

    if output_ppt:
        output_ppt_path = ft.join(output_dir, "region_detect_debug.pptx")

    if input_image is None:
        raise ValueError("No input image selected")

    # img = cv.imread(input_image)

    _, input_img_name, input_img_ext = ft.path_components(input_image)
    img = lbt.scale_image_to_video(
        image_path=input_image,
        video_path=data_video_path_for_image_scaling,
        output_path=ft.join(output_dir, input_img_name + "_scaled_to_video_size" + input_img_ext),
    )
    detector = RegionDetector(
        canny_edges_gradient=canny_edge_gradient_value,
        canny_non_edges_gradient=canny_non_edge_gradient_value,
        edge_coarse_width=edge_coarse_width_value,
    )
    regions, masks = detector.find_multiple_regions_from_interactive_points(
        image=img, debug=debug_region_detection, output_powerpoint=output_ppt_path, output_data_dir=output_dir
    )
    print("Region detection completed successfully")
