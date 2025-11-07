import os
import time
import numpy as np
from logging import DEBUG, ERROR
import cv2
import matplotlib.pyplot as plt


import opencsp.app.sofast.lib.image_processing as imgp
import opencsp.app.lookback.lookback_tools as lbt

from opencsp.common.lib.camera.Camera import Camera

import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlPointSeq as rcps
import opencsp.common.lib.render.View3d as v3d
import opencsp.common.lib.render.view_spec as vs


# Specify the folder where the log file should be saved
logger = lbt.logging_setup(
    log_folder=os.path.join(os.getcwd(), "error_logs"),
    log_file_name="error_log_lookfast_camera_adjust.txt",
    log_type=DEBUG,
)


def main():
    primary_folder = "//snl/Collaborative/NSTTF_Optics/Projects/_Directories/NSTTF_Optics_LookbackExEx/Experiments/2025-06_05_NsttfTunedFacetScan1dof/3_Post/DSC_0025"
    video_name = "DSC_0025.MOV"
    checkpoint_folder = os.path.join(primary_folder, "0_checkpoints")
    checkpoint_main_name = "lookback_main_checkpoint.json"

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

    # Arbitrary Camera Intrinsic matrix
    K = np.array([[1000, 0, 1920 / 2], [0, 1000, 1080 / 2], [0, 0, 1]])

    # Distortion coefficients
    D = np.array([0.1, -0.05, 0.001, 0.001])

    cam = Camera(intrinsic_mat=K, distortion_coef=D, image_shape_xy=tuple[1920, 1080], name="Arbitrary_Example")

    pixel_pointing = imgp.calculate_active_pixels_vectors(mask=all_pixels, camera=cam)

    reference_pixel = (670, 950)
    reference_vector = pixel_pointing[reference_pixel[0] * light_image.shape[1] + reference_pixel[1]]

    fig_control = rcfg.RenderControlFigure()
    axs_control = rca.RenderControlAxis()
    style = rcps.RenderControlPointSeq(color=None, marker=None)
    fig_rec = fm.setup_figure_for_3d_data(
        figure_control=fig_control, axis_control=axs_control, name="test", view_spec=vs.view_spec_3d()
    )
    # fig = v3d.View3d(fig_rec, axis=axs_control)
    pixel_pointing.draw_points(fig_rec, style=style)
    print("done")


if __name__ == "__main__":
    main()
