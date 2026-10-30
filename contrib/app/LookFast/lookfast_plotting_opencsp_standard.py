from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

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
import opencsp.common.lib.render.view_spec as vs
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


def make_lambda_symmetric_paraboloid_3d(
    focal_length: float,
    vertex: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    tangent: Sequence[float] = (1.0, 0.0, 0.0),
    axis_scale: float = 1.0,
    enforce_unit_vectors: bool = True,
    tol: float = 1e-8,
) -> Callable[[float, float], np.ndarray]:
    """
    Create a callable that parameterizes a *symmetric* paraboloid in 3D.

    The returned lambda maps local coordinates (u, v) in the tangent plane at the
    vertex to a 3D point on the paraboloid.

    Geometry:
      - Vertex at `vertex`
      - Axis of symmetry along unit vector `axis` (default +z)
      - One tangent direction at vertex along unit vector `tangent` (default +x)
      - The second tangent direction is constructed as axis × tangent.

    Surface equation in the local orthonormal frame {t1, t2, a}:
        w = (axis_scale) * (1 / (4*f)) * (u^2 + v^2)
    where:
        u, v are coordinates in the tangent plane,
        w is displacement along the symmetry axis,
        f is focal_length.

    Parameters
    ----------
    focal_length : float
        Paraboloid focal length. Must be > 0.
    vertex : Sequence[float], default (0,0,0)
        Vertex coordinate in 3D.
    axis : Sequence[float], default (0,0,1)
        Symmetry axis direction. Should be a unit vector if enforce_unit_vectors=True.
    tangent : Sequence[float], default (1,0,0)
        Tangent direction at the vertex. Must be orthogonal to axis.
        Should be a unit vector if enforce_unit_vectors=True.
    axis_scale : float, default 1.0
        Scales the "height" term (curvature) along the symmetry axis.
        >1 makes a "steeper" bowl; <1 makes it "flatter". Must be > 0.
    enforce_unit_vectors : bool, default True
        If True, reject non-unit vectors (within tol). If False, they will be normalized.
    tol : float, default 1e-8
        Tolerance for unit-length and orthogonality checks.

    Returns
    -------
    Callable[[float, float], np.ndarray]
        Function f(u, v) -> (x, y, z) point on the paraboloid.

    Raises
    ------
    ValueError
        On invalid focal_length/axis_scale, invalid vectors, or axis/tangent not orthogonal.
    """
    if focal_length <= 0:
        raise ValueError(f"focal_length must be > 0, got {focal_length}")
    if axis_scale <= 0:
        raise ValueError(f"axis_scale must be > 0, got {axis_scale}")

    v0 = np.asarray(vertex, dtype=float).reshape(3)
    a = np.asarray(axis, dtype=float).reshape(3)
    t = np.asarray(tangent, dtype=float).reshape(3)

    na = np.linalg.norm(a)
    nt = np.linalg.norm(t)
    if na < tol:
        raise ValueError("axis vector must be non-zero")
    if nt < tol:
        raise ValueError("tangent vector must be non-zero")

    if enforce_unit_vectors:
        if abs(na - 1.0) > tol:
            raise ValueError(f"axis must be a unit vector; ||axis||={na}")
        if abs(nt - 1.0) > tol:
            raise ValueError(f"tangent must be a unit vector; ||tangent||={nt}")
    else:
        a = a / na
        t = t / nt

    # Enforce orthogonality (tangent must lie in the tangent plane perpendicular to axis)
    dot_at = float(np.dot(a, t))
    if abs(dot_at) > tol:
        raise ValueError(
            f"axis and tangent must be orthogonal; axis·tangent={dot_at}. "
            "Provide a tangent direction perpendicular to the axis."
        )

    # Build orthonormal basis: t1 = tangent, t2 = axis × tangent, a = axis
    t1 = t
    t2 = np.cross(a, t1)
    n2 = np.linalg.norm(t2)
    if n2 < tol:
        raise ValueError("axis and tangent are (numerically) parallel or ill-conditioned; cannot form basis.")
    t2 = t2 / n2  # should already be unit if inputs are unit and orthogonal

    # Local paraboloid coefficient (matches your reference when axis=z and vertex=origin)
    a_coeff = axis_scale * (1.0 / (4.0 * focal_length))

    # return lambda u, v: v0 + u * t1 + v * t2 + (a_coeff * (u**2 + v**2)) * a # Original
    return (
        lambda x, y: v0 + x * t1 + y * t2 + (a_coeff * (x**2 + y**2)) * a
    )  # Modified to be able to symbolically call "x" and "y"


def make_lambda_symmetric_paraboloid_xy(
    focal_length: float, vertex: Sequence[float] = (0.0, 0.0, 0.0), z_scale: float = 1.0
) -> Callable[[float, float], float]:
    """
    Symmetric paraboloid opening in +z with fixed XYZ basis.

    Surface: z = z0 + z_scale * (1/(4*f)) * ((x-x0)^2 + (y-y0)^2)

    Parameters
    ----------
    focal_length : float
        Must be > 0.
    vertex : Sequence[float], default (0,0,0)
        (x0, y0, z0) vertex location.
    z_scale : float, default 1.0
        Curvature scaling along +z. Must be > 0.

    Returns
    -------
    Callable[[float, float], float]
        Function z(x, y).
    """
    if focal_length <= 0:
        raise ValueError(f"focal_length must be > 0, got {focal_length}")
    if z_scale <= 0:
        raise ValueError(f"z_scale must be > 0, got {z_scale}")

    x0, y0, z0 = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
    a = z_scale * (1.0 / (4.0 * focal_length))

    return lambda x, y: z0 + a * ((x - x0) ** 2 + (y - y0) ** 2)


def create_3_facet_heliostat():
    # Define the embedding surface (parabola) parameters
    focal_length = 4  # meters

    # Define the overall dimensions of the mirror ensemble
    total_height = 3.2  # meters
    total_width = 6.6  # meters

    # Calculate the dimensions of each facet (3 facets side by side)
    facet_width = total_width / 3  # 2.2 meters
    facet_height = total_height  # 3.2 meters

    # Define the regions for each facet (left, center, right)
    # Coordinates are in [x, y] format
    region_all_facets = RegionXY.from_vertices(
        Vxy(list(zip([1.1, 1.6], [-1.1, 1.6], [-1.1, -1.6], [1.1, -1.6])), dtype=float)
    )
    '''
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

    facet_ensemble = FacetEnsemble(facets=[left_facet, center_facet, right_facet])
    '''
    left_facet_mir = MirrorParametric(
        make_lambda_symmetric_paraboloid_xy(focal_length=focal_length, vertex=[-facet_width, 0, 0]), region_all_facets
    )
    center_facet_mir = MirrorParametric(
        make_lambda_symmetric_paraboloid_xy(focal_length=focal_length, vertex=[0, 0, 0]), region_all_facets
    )
    right_facet_mir = MirrorParametric(
        make_lambda_symmetric_paraboloid_xy(focal_length=focal_length, vertex=[facet_width, 0, 0]), region_all_facets
    )
    left_facet = Facet(name="Left", mirror=left_facet_mir)
    center_facet = Facet(name="Center", mirror=center_facet_mir)
    right_facet = Facet(name="Right", mirror=right_facet_mir)

    # Create facet ensemble
    facet_ensemble = FacetEnsemble(facets=[left_facet, center_facet, right_facet])

    # Set facet positions (centered on the embedding surface)
    facet_positions = Pxyz(
        [[facet_width, 0, -facet_width], [0, 0, 0], [0, 0, 0]]  # Left facet  # Center facet  # Right facet
    )
    facet_ensemble.set_facet_positions(facet_positions)
    iden = Rotation.identity()
    facet_ensemble.set_facet_cantings([iden, iden, iden])

    # Create heliostat
    heliostat = HeliostatAzEl(facet_ensemble, name="3-Facet Heliostat")
    # heliostat.set_orientation(heliostat.from_pointing_vector_to_configuration(Vxyz([0, 0, 1])))
    heliostat.set_orientation_from_az_el(0, np.pi / 2)

    return heliostat, facet_ensemble


def visualize_heliostat(heliostat, facet_ensemble):
    # Set up render controls
    mirror_control = RenderControlMirror(surface_normals=False, norm_len=1.0, norm_res=3)

    facet_control_left = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=True,
        draw_surface_normal=True,
        draw_name=True,
    )

    facet_control_center = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=True,
        draw_surface_normal=False,
        draw_name=True,
    )

    facet_control_right = RenderControlFacet(
        draw_mirror_curvature=True,
        mirror_styles=mirror_control,
        draw_outline=True,
        draw_surface_normal=True,
        draw_name=True,
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
        view_spec=vs.view_spec_3d(),
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
    # print("stop_here")

    # Draw the heliostat
    heliostat.draw(fig_record.view, heliostat_control)

    # Show the plot
    fig_record.view.show()
    print("stop_here")


if __name__ == "__main__":
    heliostat, facet_ensemble = create_3_facet_heliostat()
    visualize_heliostat(heliostat, facet_ensemble)
    print("done")
