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

from opencsp.common.lib.csp.FacetEnsemble import FacetEnsemble
from opencsp.common.lib.csp.Facet import Facet
from opencsp.common.lib.csp.MirrorParametric import MirrorParametric
from opencsp.common.lib.csp.HeliostatAzEl import HeliostatAzEl
from opencsp.common.lib.geometry.RegionXY import RegionXY
from opencsp.common.lib.geometry.Vxy import Vxy
from opencsp.common.lib.geometry.Pxyz import Pxyz
from opencsp.common.lib.geometry.TransformXYZ import TransformXYZ
from opencsp.common.lib.render.View3d import View3d
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
from opencsp.common.lib.render_control.RenderControlFacetEnsemble import RenderControlFacetEnsemble
from opencsp.common.lib.render_control.RenderControlFacet import RenderControlFacet
from opencsp.common.lib.render_control.RenderControlMirror import RenderControlMirror
from opencsp.common.lib.render_control.RenderControlHeliostat import RenderControlHeliostat

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


def create_3_facet_heliostat():
    # Define the embedding surface (parabola) parameters
    focal_length = 66.0  # meters

    # Define the overall dimensions of the mirror ensemble
    total_height = 3.2  # meters
    total_width = 6.6  # meters

    # Calculate the dimensions of each facet (3 facets side by side)
    facet_width = total_width / 3  # 2.2 meters
    facet_height = total_height  # 3.2 meters

    # Define the regions for each facet (left, center, right)
    # Coordinates are in [x, y] format
    left_region = RegionXY.from_vertices(
        Vxy(list(zip([-1.1, 1.6], [-3.3, 1.6], [-3.3, -1.6], [-1.1, -1.6])), dtype=float)
    )
    center_region = RegionXY.from_vertices(
        Vxy(list(zip([1.1, 1.6], [-1.1, 1.6], [-1.1, -1.6], [1.1, -1.6])), dtype=float)
    )
    right_region = RegionXY.from_vertices(Vxy(list(zip([3.3, 1.6], [1.1, 1.6], [1.1, -1.6], [3.3, -1.6])), dtype=float))

    # Create mirrors for each facet
    left_mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, left_region)
    center_mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, center_region)
    right_mirror = MirrorParametric.generate_symmetric_paraboloid(focal_length, right_region)

    # Create facets
    left_facet = Facet(name="Left", mirror=left_mirror)
    center_facet = Facet(name="Center", mirror=center_mirror)
    right_facet = Facet(name="Right", mirror=right_mirror)

    # Create facet ensemble
    facet_ensemble = FacetEnsemble(facets=[left_facet, center_facet, right_facet])

    # Set facet positions (centered on the embedding surface)
    facet_positions = Pxyz(
        [[-facet_width, 0, facet_width], [0, 0, 0], [0, 0, 0]]  # Left facet  # Center facet  # Right facet
    )
    # facet_ensemble.set_facet_positions(facet_positions)

    # Create heliostat
    heliostat = HeliostatAzEl(facet_ensemble, name="3-Facet Heliostat")
    # heliostat.default_direction()
    # heliostat.set_orientation(heliostat.from_pointing_vector_to_configuration(Vxyz([0, 0, 1])))

    return heliostat, facet_ensemble


def visualize_heliostat(heliostat, facet_ensemble):
    # Set up render controls
    mirror_control = RenderControlMirror(surface_normals=False, norm_len=1.0, norm_res=3)

    facet_control_left = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=False,
        draw_surface_normal=True,
        draw_name=False,
    )

    facet_control_center = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=False,
        draw_surface_normal=False,
        draw_name=False,
    )

    facet_control_right = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=True,
        draw_surface_normal=True,
        draw_name=False,
    )

    facet_ensemble_control = RenderControlFacetEnsemble(
        default_style=facet_control_center,
        draw_outline=True,
        draw_centroid=True,
        draw_normal_vector=True,
        special_styles={"Left": facet_control_left, "Center": facet_control_center, "Right": facet_control_right},
    )

    heliostat_control = RenderControlHeliostat(
        draw_centroid=False, facet_ensemble_style=facet_ensemble_control, draw_facet_ensemble=True
    )

    fig_control = rcfg.RenderControlFigure()

    # Set up figure
    fig_record = fm.setup_figure_for_3d_data(
        figure_control=fig_control,
        axis_control=None,
        view_spec=None,
        equal=True,
        number_in_name=False,
        input_prefix=None,
        name="3_facet_heliostat",
        title="3-Facets (3.2m × 6.6m)",
        caption="Three-facet heliostat with parabolic mirrors (f=66m)",
        comments=["Each facet is 3.2m tall × 2.2m wide"],
        code_tag=None,
    )

    # Draw the facet ensemble
    facet_ensemble.draw(fig_record.view, facet_ensemble_style=facet_ensemble_control, transform=None)
    # Show the plot
    fig_record.view.show()
    print("stop_here")

    # Draw the heliostat
    heliostat.draw(fig_record.view, heliostat_control)

    # Show the plot
    fig_record.view.show()
    print("stop_here")


if __name__ == "__main__":
    heliostat, facet_ensemble = create_3_facet_heliostat()
    visualize_heliostat(heliostat, facet_ensemble)
    print("done")
