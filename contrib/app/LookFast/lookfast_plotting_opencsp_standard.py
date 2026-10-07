import numpy as np
import cv2

from opencsp.common.lib.csp.StandardPlotOutput import StandardPlotOutput
from opencsp.common.lib.csp.LightSourceSun import LightSourceSun
import opencsp.common.lib.tool.file_tools as ft
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Vxyz import Vxyz
from opencsp.common.lib.geometry.Uxyz import Uxyz
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.LoopXY import LoopXY
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.csp.MirrorPoint import MirrorPoint
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric

# import opencsp.app.lookback.lookback_tools as lbt
import contrib.app.LookFast.lookback_tools as lbt


def opencsp_standard_plotting(
    output_directory, solution_set, plot_configuration_file, ideal_and_real_mirror_configuration_file
):

    ft.create_directories_if_necessary(output_directory)

    coords_centroid = np.mean(np.array(solution_set["coords"]), axis=0)
    centered_coords = np.array(solution_set["coords"]) - coords_centroid
    coords_centroid = np.mean(centered_coords, axis=0)

    rect = cv2.minAreaRect(np.float32(centered_coords[:, 0:2]))
    box = cv2.boxPoints(rect)  # In test, points of box came out in CCW starting with the top left

    expected_facet_corners = Vxy(
        list(zip(box[3], box[2], box[1], box[0])), dtype=float
    )  # Counterclockwise Starting from Top Right Corner in [row, column] SOFAST Example uses this convention

    loop = LoopXY.from_vertices(expected_facet_corners)
    region = RegionXY(loop)

    # Get measured and reference optics
    # mirror_measured = sofast.get_optic().mirror.no_parent_copy()
    mirror_measured = MirrorPoint(
        surface_points=Pxyz(centered_coords.T),
        normal_vectors=Uxyz(solution_set["slopes"].T),
        shape=region,
        interpolation_type="nearest",
    )

    mirror_reference = MirrorParametric.generate_symmetric_paraboloid(100, mirror_measured.region)

    plots = StandardPlotOutput()
    plots.set_plot_control_from_settings(plot_configuration_file)
    plots.plot()
