import cv2 as cv
from PIL import Image
import numpy as np
import json

import opencsp.common.lib.cv.image_reshapers as ir
from opencsp.common.lib.cv.spot_analysis.image_processor import *
import opencsp.common.lib.geometry.LineXY as l2
import opencsp.common.lib.geometry.Pxy as p2
import contrib.common.lib.geometry.RectXY as r2
import opencsp.common.lib.geometry.RegionXY as reg2
import opencsp.common.lib.render.Color as color
import opencsp.common.lib.render.figure_management as fm
import opencsp.common.lib.render.PowerpointSlide as pps
import opencsp.common.lib.render.view_spec as vs
import opencsp.common.lib.render_control.RenderControlAxis as rca
import opencsp.common.lib.render_control.RenderControlFigure as rcfg
import opencsp.common.lib.render_control.RenderControlPowerpointPresentation as rcpp
import opencsp.common.lib.render_control.RenderControlPowerpointSlide as rcps
import opencsp.common.lib.tool.file_tools as ft
import opencsp.common.lib.tool.log_tools as lt

import contrib.app.LookFast.lookback_tools as lbt


class RegionDetector:
    """
    A class used to detect regions in images using various image processing
    techniques including Canny edge detection, blob analysis, ray intersection,
    edge identification, and edge intersection.

    Attributes
    ----------
    edge_coarse_width : int
        The maximum encapsulating width of the edges for :py:meth:`step5_find_edge_groups`.
    canny_edges_gradient : int
        The threshold gradient value for edges in Canny edge detection.
    canny_non_edges_gradient : int
        The threshold gradient value for non-edges in Canny edge detection.
    canny_test_gradients : list[tuple[int, int]]
        A list of tuples containing (edge, non-edge) gradient pairs to test for
        Canny edge detection. Useful for setting up edge detection with new
        experiments.
    edge_colors : dict
        What colors to use to indicate the "top", "right", "bottom", and "left" edges.
    corner_colors : dict
        What colors to use to indicate the "tl", "tr", "br", and "bl" corners.
    images_to_show : list[tuple[str, np.ndarray]]
        A list of images to be displayed for use with canny_test_gradients.
    summary_visualizations : list[tuple[str, np.ndarray]]
        A list of summary visualizations that explain the steps used in locating
        the region.
    ppt_deck : RenderControlPowerpointPresentation
        The PowerPoint presentation control.
    slide_control : RenderControlPowerpointSlide
        The PowerPoint slide control.
    generate_powerpoint: bool
        Flag indicating whether to generate a PowerPoint presentation from the detected regions.
    algimages_canny: dict[str, np.ndarray]
        Dictionary containing Canny edge detection images for each region,
        where the key is the region name and the value is the corresponding image.
    algimages_blob_analysis: dict[str, np.ndarray]
        Dictionary containing blob analysis images for each region,
        where the key is the region name and the value is the corresponding image.
    algimages_captions: dict[str, np.ndarray]
        Dictionary containing caption images for each region,
        where the key is the region name and the value is the corresponding image.
    algimages_boundary_edges: dict[str, np.ndarray]
        Dictionary containing boundary edge images for each region,
        where the key is the region name and the value is the corresponding image.
    algimages_find_edges: dict[str, np.ndarray]
        Dictionary containing edge detection images for each region,
        where the key is the region name and the value is the corresponding image.
    algimages_assign_edges_and_corners: dict[str, np.ndarray]
        Dictionary containing images with assigned edges and corners for each region,
        where the key is the region name and the value is the corresponding image.
    """

    def __init__(
        self,
        edge_coarse_width: int,
        canny_edges_gradient=10,
        canny_non_edges_gradient=5,
        canny_test_gradients: list[tuple[int, int]] = None,
        generate_powerpoint: bool = False,
    ):
        """
        Parameters
        ----------
        edge_coarse_width : int
            The maximum encapsulating width of the edges for
            :py:meth:`step5_find_edge_groups`. This should at least as large as
            width of the largest vertical edge, or the height of the largest
            horizontal edge.
        canny_edges_gradient : int, optional
            The threshold gradient value for edges in Canny edge detection.
        canny_non_edges_gradient : int, optional
            The threshold gradient value for non-edges in Canny edge detection.
        canny_test_gradients : list[tuple[int, int]], optional
            A list of tuples containing (edge, non-edge) gradient pairs to test
            for Canny edge detection. Useful for setting up edge detection with
            new experiments.  For example: [(10,5), (10,10), (10,15), (20,5),
            (20,10), (20,15)].  Default None.
        generate_powerpoint : bool, optional
            If True, then this instance will generate a powerpoint with all
            the algorithm images "algimages_*" in it. Default is False.
        """
        self.edge_coarse_width = edge_coarse_width
        self.canny_edges_gradient = canny_edges_gradient
        self.canny_non_edges_gradient = canny_non_edges_gradient
        self.canny_test_gradients = canny_test_gradients
        self.generate_powerpoint = generate_powerpoint

        # visualization values
        self.edge_colors = {
            "top": color.red(),
            "right": color.cyan(),
            "bottom": color.magenta(),
            "left": color.yellow(),
        }
        self.corner_colors = {"tl": color.red(), "tr": color.cyan(), "br": color.magenta(), "bl": color.yellow()}
        self.images_to_show: list[tuple[str, np.ndarray]] = []
        self.summary_visualizations: list[tuple[str, np.ndarray]] = []

        # debugging/algorithm images
        self.algimages_canny: dict[str, np.ndarray] = None
        self.algimages_blob_analysis: dict[str, np.ndarray] = None
        self.algimages_captions: dict[str, np.ndarray] = None
        self.algimages_boundary_edges: dict[str, np.ndarray] = None
        self.algimages_find_edges: dict[str, np.ndarray] = None
        self.algimages_assign_edges_and_corners: dict[str, np.ndarray] = None

        # powerpoint values
        self.ppt_deck: rcpp.RenderControlPowerpointPresentation = None
        self.slide_control: rcps.RenderControlPowerpointSlide = None

    def draw_image(self, title: str, image: np.ndarray, show=False, block=False):
        """
        Draws an image and optionally displays it.

        Parameters
        ----------
        title : str
            The title of the image.
        image : np.ndarray
            The image to be drawn.
        show : bool, optional
            Whether to show the image immediately, by default False.
        block : bool, optional
            Whether to block the execution until the image window is closed, by default False.
        """
        self.images_to_show.append((title, image.copy()))

        if show:
            rows = 1
            if len(self.images_to_show) > 3:
                rows = 2
            if len(self.images_to_show) > 8:
                rows = 3
            cols = int(np.ceil(len(self.images_to_show) / rows))

            axis_control = rca.image(grid=False)
            figure_control = rcfg.RenderControlFigure(tile=True, tile_array=(cols, rows))
            view_spec_2d = vs.view_spec_im()
            fig_record = None
            for title, img in self.images_to_show:
                if fig_record is not None:
                    fig_record.view.show(block=False)
                fig_record = fm.setup_figure(
                    figure_control, axis_control, view_spec_2d, title=title, code_tag=f"{__file__}", equal=False
                )
                fig_record.view.imshow(Image.fromarray(img))
            if block:
                lt.info("Showing images to user and waiting...")
            fig_record.view.show(block=block)
            if block:
                fig_record.close()
                lt.info("...done")
            self.images_to_show.clear()

    def draw_images(self, images: dict[str, np.ndarray] | list[tuple[str, np.ndarray]]):
        """
        Draws multiple images and displays them.

        Parameters
        ----------
        images : list[tuple[str, np.ndarray]]
            A list of tuples containing the title and image to be drawn.
        """
        if isinstance(images, dict):
            images = [(title, image) for title, image in images.items()]

        for i, title_image in enumerate(images):
            show = i == len(images) - 1
            self.draw_image(title_image[0], title_image[1], show, show)

    def step1_canny_edge_detection(self, debug_canny_settings: bool, image: np.ndarray) -> np.ndarray:
        """
        Performs Canny edge detection on the given image.

        Parameters
        ----------
        debug_canny_settings : bool
            If True, or if canny_test_gradients is set, then debug images will be shown.
        image : np.ndarray
            The input image on which to perform edge detection.

        Returns
        -------
        np.ndarray
            A boolean array with the same width/height as the input image, where
            1s represent pixels that are classified as edges and 0s represent
            non-edges.
        """
        self.algimages_canny = {}
        self.algimages_canny["input image"] = image

        if self.canny_test_gradients is not None:
            for i, gradients in enumerate(self.canny_test_gradients):
                # Exploring parameters for canny edge detection
                edges_gradient, non_edges_gradient = gradients

                canny: np.ndarray = cv.Canny(image, non_edges_gradient, edges_gradient)

                image_title = f"Canny (edges gradient {edges_gradient}, non-edges gradient {non_edges_gradient})"
                if edges_gradient == self.canny_edges_gradient and non_edges_gradient == self.canny_non_edges_gradient:
                    image_title += "**"
                self.algimages_canny[image_title] = canny

        # chosen canny parameters
        canny_edges: np.ndarray = cv.Canny(image, self.canny_non_edges_gradient, self.canny_edges_gradient)
        image_title = (
            f"Canny (edges gradient {self.canny_edges_gradient}, non-edges gradient {self.canny_non_edges_gradient})**"
        )
        self.summary_visualizations.append((image_title, canny_edges))

        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=2, ncols=3, slide_control=self.slide_control)
            slide.set_title("step1_canny_edge_detection")
            for caption, image in self.algimages_canny.items():
                image = ir.false_color_reshaper(image).astype(np.uint8)
                slide.add_image(pps.PowerpointImage(image, caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_canny_settings or self.canny_test_gradients:
            self.draw_images(self.algimages_canny)

        return canny_edges

    def step2_blob_analysis(
        self, canny_edges: np.ndarray, debug_blob_analysis=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies horizontal and vertical blob growth/filtering to the canny edges.

        Parameters
        ----------
        canny_edges : np.ndarray[uint8]
            The edge pixels as found with :py:meth:`step1_canny_edge_detection`.
        debug_blob_analysis : bool, optional
            True to show debug images, by default False.

        Returns
        -------
        horizontal_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where the horizontal edge pixels are detected.
        vertical_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where the vertical edge pixels are detected.
        negative_mask : np.ndarray[bool]
            Numpy array with the same width and height as the given canny_edges
            image. Non-zero where there aren't any edges.
        """
        slice_l, slice_s = 13, 3  # long, short
        slice_area = slice_l * slice_s  # 13*3 = 39
        slice_thresh = 10 / slice_area  # 10/39 = 0.26
        horizontal_kernel = np.ones((slice_s, slice_l), np.float32) / slice_area
        vertical_kernel = np.ones((slice_l, slice_s), np.float32) / slice_area

        # convolve with the horizontal kernel
        binary_edges = np.clip(canny_edges, 0, 1).astype(np.uint8)
        negative_horizontal = binary_edges.copy().astype(np.float64)
        negative_horizontal: np.ndarray = cv.filter2D(src=negative_horizontal, ddepth=-1, kernel=horizontal_kernel)
        negative_horizontal[np.where(negative_horizontal < slice_thresh)] = 0
        negative_horizontal[np.where(negative_horizontal > 0)] = 255
        horizontal_mask = np.full(negative_horizontal.shape, False)
        horizontal_mask[np.where(negative_horizontal != 0)] = True

        # convolve with the vertical kernel
        negative_vertical = binary_edges.copy().astype(np.float64)
        negative_vertical: np.ndarray = cv.filter2D(src=negative_vertical, ddepth=-1, kernel=vertical_kernel)
        negative_vertical[np.where(negative_vertical < slice_thresh)] = 0
        negative_vertical[np.where(negative_vertical > 0)] = 255
        vertical_mask = np.full(negative_vertical.shape, False)
        vertical_mask[np.where(negative_vertical != 0)] = True

        negative_mask = np.logical_not(np.logical_or(horizontal_mask, vertical_mask))
        thin_blob = canny_edges.copy()
        thin_blob[negative_mask] = 0
        self.summary_visualizations.append(("Horizontal+Vertical Blob", thin_blob))

        fat_blob = thin_blob.copy().astype(np.float64) / 255
        fat_blob: np.ndarray = cv.filter2D(fat_blob, -1, np.ones((7, 7), np.float32))
        fat_blob[np.where(fat_blob > 0.5)] = 255

        self.algimages_blob_analysis = {
            "canny_edges": canny_edges,
            "horizontal_mask": horizontal_mask,
            "vertical_mask": vertical_mask,
            "thin_blob": thin_blob,
            "fat_blob": fat_blob,
        }
        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=2, ncols=3, slide_control=self.slide_control)
            slide.set_title("step2_blob_analysis")
            for caption, image in self.algimages_blob_analysis.items():
                image = ir.false_color_reshaper(image).astype(np.uint8)
                slide.add_image(pps.PowerpointImage(image, caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_blob_analysis:
            self.draw_images(self.algimages_blob_analysis)

        return horizontal_mask, vertical_mask, negative_mask

    def step3_ray_project_edge_intercept(
        self, thin_boundaries: np.ndarray, approx_area_center_pixel: p2.Pxy, debug_ray_projection=False
    ) -> tuple[p2.Pxy, p2.Pxy]:
        """
        Project rays out from the given approximate center to find the first
        intercepts with the boundaries image, if any. This process is iterative,
        and updates the center approximation at each step.

        Parameters
        ----------
        thin_boundaries : np.ndarray[uint8]
            The boundaries image to test against. Should be 0 where there aren't
            any boundaries, or non-0 where there are boundaries.
        approx_area_center_pixel : p2.Pxy
            Where to start ray projecting from.
        debug_ray_projection : bool, optional
            True to show debug image and block, by default False

        Returns
        -------
        area_center: p2.Pxy
            The updated approximation of the area's center.
        boundary_samples: p2.Pxy
            The set of sampled boundary intercepts, minus those that fall outside the typical range.
        """
        niterations = 10
        nrays = 50

        # initialize/declare variables
        all_boundary_samples: p2.Pxy = None
        """ All the sampled points """
        boundary_samples: p2.Pxy = None
        """ All the sampled points that are within the typical distance """
        area_center = p2.Pxy(approx_area_center_pixel)
        """ Approximate center of the area. Updated each iteration. """
        all_area_centers: list[p2.Pxy] = []
        """ All area_center values, for debugging """
        self.algimages_captions = {}

        # get the bounding box, to make calculating rays easier
        image_height, image_width = thin_boundaries.shape[0], thin_boundaries.shape[1]
        image_vertices = [[0, 0], [image_width, 0], [image_width, image_height], [0, image_height]]
        image_vertices = [p2.Pxy(xy_pair) for xy_pair in image_vertices]
        image_area = reg2.RegionXY.from_vertices(p2.Pxy.from_list(image_vertices))

        # get a base image that we can additively draw to
        sampling_image = thin_boundaries.copy()
        if sampling_image.ndim > 2:
            sampling_image = sampling_image[:, :, 0]
        sampling_image[np.where(sampling_image != 0)] = 125
        debug_all_image = sampling_image.copy()

        # For finding the closes point in the given list to the given reference
        def get_closest_point(reference: p2.Pxy, points_list: p2.Pxy):
            deltas = p2.Pxy([points_list.x - reference.x[0], points_list.y - reference.y[0]])
            distances = deltas.magnitude()
            min_idx = np.argmin(distances)
            min_pnt = p2.Pxy((points_list.x[min_idx], points_list.y[min_idx]))
            return min_pnt

        # project some rays
        for iteration in range(niterations):
            iteration_samples: list[p2.Pxy] = []
            debug_image = sampling_image.copy()

            for ray_idx in range(nrays):
                angle = np.random.random() * np.pi * 2  # radians
                ray = l2.LineXY.from_location_angle(area_center, angle)
                pnt1, pnt2 = None, None

                # draw the line
                image_edge_intersect = image_area.loops[0].intersect_line(ray)
                iei1 = (int(image_edge_intersect.x[0]), int(image_edge_intersect.y[0]))
                iei2 = (int(image_edge_intersect.x[1]), int(image_edge_intersect.y[1]))
                ray_image = cv.line(np.zeros_like(sampling_image), iei1, iei2, color=(125), thickness=2)

                # add the line's values to the sampling_image's values
                # get the intersection locations
                ray_image += sampling_image
                y_intersects, x_intersects = np.where(ray_image == 250)
                if len(x_intersects) > 0:
                    xy_intersects = p2.Pxy([x_intersects, y_intersects])

                    # find the closest point to the center
                    pnt1 = get_closest_point(area_center, xy_intersects)
                    iteration_samples.append(pnt1)

                    # remove all points on this side of the line
                    bisect = l2.LineXY.from_location_angle(area_center, angle + (np.pi / 2))
                    if not (bisect.dist_from_signed(pnt1)[0] < 0):
                        bisect = bisect.flip()
                    intersect_is_other_side = bisect.dist_from_signed(xy_intersects) > 0
                    other_half_xy_intersects = xy_intersects[np.where(intersect_is_other_side)]

                    # find the other closest point to the center, on the other side
                    if len(other_half_xy_intersects) > 0:
                        pnt2 = get_closest_point(area_center, other_half_xy_intersects)
                        iteration_samples.append(pnt2)

                # debugging
                center_pixel = (int(area_center.x[0]), int(area_center.y[0]))
                pnt1_pixel, pnt2_pixel = iei1, iei2
                if pnt1 is not None:
                    pnt1_pixel = (int(pnt1.x[0]), int(pnt1.y[0]))
                if pnt2 is not None:
                    pnt2_pixel = (int(pnt2.x[0]), int(pnt2.y[0]))
                ray_image = cv.circle(ray_image, center_pixel, 7, color=(255))
                debug_image = cv.line(debug_image, pnt1_pixel, pnt2_pixel, color=(50), thickness=2)
                debug_image = cv.circle(debug_image, center_pixel, 4, color=(255))
                debug_all_image = cv.line(debug_all_image, pnt1_pixel, pnt2_pixel, color=(50), lineType=cv.LINE_AA)
                if pnt1 is not None:
                    ray_image = cv.circle(ray_image, pnt1_pixel, 7, color=(255))
                    debug_image = cv.circle(debug_image, pnt1_pixel, 4, color=(255))
                    debug_all_image = cv.drawMarker(debug_all_image, pnt1_pixel, color=(255), markerSize=1)
                if pnt2 is not None:
                    ray_image = cv.circle(ray_image, pnt2_pixel, 7, color=(255))
                    debug_image = cv.circle(debug_image, pnt2_pixel, 4, color=(255))
                    debug_all_image = cv.drawMarker(debug_all_image, pnt2_pixel, color=(255), markerSize=1)
                # self.draw_image(f"Ray {ray_idx} ({center_pixel=}, deg={np.rad2deg(angle)})", debug_image, True, True)

            # record information from this iteration
            self.algimages_captions[f"Ray projection, iteration {iteration}"] = debug_image
            if all_boundary_samples is None:
                all_boundary_samples = p2.Pxy.from_list(iteration_samples)
                boundary_samples = p2.Pxy(all_boundary_samples.data)
            else:
                all_boundary_samples = p2.Pxy.from_list([all_boundary_samples] + iteration_samples)

            # discard duplicate points
            all_boundary_samples_xy = list(zip(all_boundary_samples.x.tolist(), all_boundary_samples.y.tolist()))
            unique_boundary_samples = list(set(all_boundary_samples_xy))
            unique_samples_x = [v[0] for v in unique_boundary_samples]
            unique_samples_y = [v[1] for v in unique_boundary_samples]
            all_boundary_samples = p2.Pxy([unique_samples_x, unique_samples_y])

            # choose the samples that aren't outliers
            deltas = p2.Pxy([all_boundary_samples.x - area_center.x[0], all_boundary_samples.y - area_center.y[0]])
            distances = deltas.magnitude()
            distances_list = sorted(distances.tolist())
            mid50 = distances_list[int(len(distances) * 0.25) : int(len(distances) * 0.75)]
            average_distance = np.average(mid50)
            in_range = np.logical_and(distances > average_distance / 2, distances < average_distance * 1.5)
            boundary_samples = all_boundary_samples[np.where(in_range)]
            lt.info(f"\rSampled {len(boundary_samples)} points", end="")

            # update center pixel approximation
            all_area_centers.append(area_center)
            area_center = p2.Pxy(np.average(boundary_samples.data, axis=1))

        lt.info("")

        # draw the debug_all_image with all rays and all sampled points
        for ac in all_area_centers:
            center_pixel = (int(ac.x[0]), int(ac.y[0]))
            debug_all_image = cv.drawMarker(debug_all_image, center_pixel, color=(255), markerSize=3)
        self.algimages_captions[f"Ray projection, sum"] = debug_all_image
        self.summary_visualizations.append((f"Ray Projection", ir.false_color_reshaper(debug_all_image)))

        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=3, ncols=4, slide_control=self.slide_control)
            slide.set_title("step3_ray_project_edge_intercept")
            for caption, image in self.algimages_captions.items():
                image = ir.false_color_reshaper(image).astype(np.uint8)
                slide.add_image(pps.PowerpointImage(image, caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_ray_projection:
            self.draw_images(self.algimages_captions)

        return area_center, boundary_samples

    def step4_boundary_locations_to_edges(
        self, boundary_locations: p2.Pxy, canny_edges: np.ndarray, diameter=7, debug_ray_projection=False
    ) -> np.ndarray:
        """
        Get all edge pixels within a small distance of the boundary locations.

        Parameters
        ----------
        boundary_locations : p2.Pxy
            The list of boundary locations to match against.
        canny_edges : np.ndarray[uint8]
            All edge pixels, as from the canny edge detector.
        diameter : int, optional
            The distance that an edge pixel can be from the boundary locations.
            Applies a box filter of this width and height.

        Returns
        -------
        boundary_edges: np.ndarray[uint8]
            The edge pixels close to the boundary locations. Matched edge pixels
            will be 255, all others will be 0.
        """
        diameter = 7

        boundary_pixels = np.zeros_like(canny_edges)
        boundary_pixels[boundary_locations.asindex("yx")] = 255

        boundary_blob = boundary_pixels.copy().astype(np.float64)
        boundary_blob: np.ndarray = cv.filter2D(boundary_blob, -1, np.ones((diameter, diameter), np.float32))

        matched_edges = canny_edges.copy()
        matched_edges[np.where(boundary_blob == 0)] = 0

        self.algimages_boundary_edges = {
            f"Boundary pixels": boundary_pixels,
            f"Boundary blob": boundary_blob,
            f"Boundary edges": matched_edges,
        }
        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=3, slide_control=self.slide_control)
            slide.set_title("step4_boundary_locations_to_edges")
            for caption, image in self.algimages_boundary_edges.items():
                image = ir.false_color_reshaper(image).astype(np.uint8)
                slide.add_image(pps.PowerpointImage(image, caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_ray_projection:
            self.draw_images(self.algimages_boundary_edges)

        return matched_edges

    def step5_find_edge_groups(
        self, boundary_edges: np.ndarray, debug_edge_groups=False
    ) -> tuple[list[r2.RectXY], np.ndarray]:
        """
        Finds the top/bottom and left/right groups of edge pixels in the given
        edges image. We do this by looking for large groups of pixels in the x
        and y axes.

        Two iterations are done. In the first iteration the rough location of
        the edges is determined by scanning all edge pixels. In the second
        iteration, the search region is constrained to only include the region
        where we expect the edges to be, and to exclude the corners.

        Parameters
        ----------
        boundary_edges : np.ndarray[uint8]
            A black and white image to search for edges within, with 0 pixels
            (non-edges) and non-zero pixels (edges).

        Returns
        -------
        edge_groups : tuple[list[r2.RectXY], np.ndarray]
            The four edge groups
        """
        refined_edge_groups: list[r2.RectXY] = []
        width = boundary_edges.shape[1]
        height = boundary_edges.shape[0]
        self.algimages_find_edges = {}

        # find initial edge groups, two per 'x' and 'y' search dir2.rections
        scan_dirs = ("y", "x")
        scansteps_xy = (np.array([0, 1]), np.array([1, 0]))
        windows_wh = (np.array([width, self.edge_coarse_width]), np.array([self.edge_coarse_width, height]))
        initial_edge_groups: dict[str, tuple[r2.RectXY, r2.RectXY]] = {}
        for scan_dir, scanstep_xy, window_wh in zip(scan_dirs, scansteps_xy, windows_wh):
            initial_edge_groups[scan_dir] = self._find_edge_groups_xory(
                boundary_edges, scan_dir, scanstep_xy, window_wh
            )

        # find the edge groups again, excluding pixels from the opposite groups
        for scan_dir, scanstep_xy, window_wh in zip(scan_dirs, scansteps_xy, windows_wh):
            # get the subregion to scan within
            if scan_dir == "y":
                groups = initial_edge_groups["x"]
                left, right = sorted(groups, key=lambda g: g.left)
                region = r2.RectXY(p2.Pxy((left.right, 0)), p2.Pxy((right.left, height)))
                window_wh = np.array([region.right - region.left, self.edge_coarse_width])
            elif scan_dir == "x":
                groups = initial_edge_groups["y"]
                top, bottom = sorted(groups, key=lambda g: g.top)
                region = r2.RectXY(p2.Pxy((0, top.bottom)), p2.Pxy((width, bottom.top)))
                window_wh = np.array([self.edge_coarse_width, region.bottom - region.top])
            boundary_edges_region = boundary_edges[
                int(region.top) : int(region.bottom), int(region.left) : int(region.right)
            ]

            # get a more accurate scan, now that the confusing pixels at the corners are excluded
            offset_edge_groups = self._find_edge_groups_xory(boundary_edges_region, scan_dir, scanstep_xy, window_wh)
            for edge_group in offset_edge_groups:
                refined_edge_groups.append(edge_group + region.tl)

        # visualize the found edge groups
        vis_image = np.expand_dims(boundary_edges, axis=2)
        vis_image = np.broadcast_to(vis_image, (vis_image.shape[0], vis_image.shape[1], 3))
        captions_groups = [
            ("Initial edge groups", initial_edge_groups["x"] + initial_edge_groups["y"]),
            ("Refined edge groups", refined_edge_groups),
        ]
        for caption, edge_groups in captions_groups:
            edge_group_image = vis_image.copy()

            for edge_group in edge_groups:
                tl, br = (int(edge_group.left), int(edge_group.top)), (int(edge_group.right), int(edge_group.bottom))
                edge_group_image = cv.rectangle(edge_group_image, tl, br, color.magenta().rgb_255(), thickness=2)

            self.algimages_find_edges[caption] = edge_group_image
        self.summary_visualizations.append(("Edge Groups", self.algimages_find_edges["Refined edge groups"]))

        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=2, slide_control=self.slide_control)
            slide.set_title("step5_find_edge_groups")
            for caption, image in self.algimages_find_edges.items():
                slide.add_image(pps.PowerpointImage(image.astype(np.uint8), caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_edge_groups:
            self.draw_images(self.algimages_find_edges)

        return refined_edge_groups

    def _find_edge_groups_xory(
        self, canny: np.ndarray, scan_dir: str, scanstep_xy: np.ndarray, window_wh: np.ndarray
    ) -> tuple[r2.RectXY, r2.RectXY]:
        """
        Finds clumps of pixels along the x or y axis. A sliding window approach
        is used with size window_wh, and has a resolution of scanstep_xy. Two
        mutually exclusive windows are returned for the windows with the largest
        number of edge pixels.

        Parameters
        ----------
        canny : np.ndarray[number]
            Image with non-edge (0) and edge (non-zero) pixels.
        scan_dir : str
            The dir2.rection to scan in. One of 'x' or 'y'.
        scanstep_xy : np.ndarray[int]
            The resolution of the sliding window. Should be a length-2 numpy array.
        window_wh : np.ndarray[int]
            The size of the window. Should be a length-2 numpy array. Either the
            width should match the width of the given canny image, or the height
            should match.

        Returns
        -------
        tuple[r2.RectXY, r2.RectXY]
            The two largest windows.
        """
        ret: list[r2.RectXY] = []
        width = canny.shape[1]
        height = canny.shape[0]
        previous_window: r2.RectXY = None
        scanstep_xy = p2.Pxy(scanstep_xy)
        window_wh = p2.Pxy(window_wh)

        # validate input
        if window_wh.x[0] != width:
            if window_wh.y[0] != height:
                lt.error_and_raise(
                    ValueError,
                    "Error in TargetBoardLocatorImageProcessor._find_edge_groups_xory(): "
                    + f"either the width or height of window_wh should match the given image, "
                    + "but {window_wh.x[0]=} != {width=} and {window_wh.y[0]=} != {height=}",
                )

        # search for the two largest windows
        for i in range(2):
            # position of the window to search within
            curr_pos = r2.RectXY(p2.Pxy((0, 0)), window_wh) - scanstep_xy
            largest_pos: r2.RectXY = None
            largest_count = -1

            while True:
                curr_pos += scanstep_xy
                if (curr_pos.bottom >= height) and (curr_pos.right >= width):
                    break

                # We want two well-separated lines for the top/bottom and
                # left/right sides. Don't check within the overlapping
                # region of the previously found window.
                if i == 1:
                    if scan_dir == "y":
                        if (curr_pos.bottom > previous_window.top) and (curr_pos.top < previous_window.bottom):
                            continue
                    elif scan_dir == "x":
                        if (curr_pos.right > previous_window.left) and (curr_pos.left < previous_window.right):
                            continue
                    else:
                        raise RuntimeError

                # get the current window
                window = canny[int(curr_pos.top) : int(curr_pos.bottom), int(curr_pos.left) : int(curr_pos.right)]

                # check if this is the largest window
                count = np.sum(window)
                if count > largest_count:
                    largest_pos = curr_pos
                    largest_count = count

            # register the location of the largest window as one of our edge groups
            ret.append(largest_pos)

            # get ready for the next iteration in the same scan direction
            previous_window = largest_pos

        return ret

    def step6_assign_edges_and_corners(
        self, edge_groups: list[r2.RectXY], canny: np.ndarray, debug_edge_assignment=False
    ) -> tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Finds the edges in the canny image for each edge group, and assign the
        edges to be the "left", "top", "right" or "bottom" of a rectangle. Uses
        the RANSAC algorithm to find the edges.

        Parameters
        ----------
        edge_groups : list[XY]
            List of regions to search for edges within using RANSAC. Should be
            length 4, with one group for each of the top, right, bottom, and
            left sides. Order doesn't matter.
        canny : np.ndarray
            The edges image from the canny algorithm.
        debug_edge_assignment : bool, optional
            True to draw debug images, by default False

        Returns
        -------
        edges: dict[str, l2.LineXY]
            One edge per "left", "top", "right", and "bottom"
        corners: dict[str, p2.Pxy]
            The intersection points of the edges. Includes "tl", "tr", "br", and "bl".
        region: reg2.RegionXY
            The region described the by the corners.
        """

        edges: dict[str, l2.LineXY] = {"top": None, "bottom": None, "left": None, "right": None}
        corners: dict[str, p2.Pxy] = {"tl": None, "tr": None, "br": None, "bl": None}
        region: reg2.RegionXY = None
        width, height = canny.shape[1], canny.shape[0]
        self.algimages_assign_edges_and_corners = {}

        # use RANSAC to assign lines to the edge groups
        lines: list[l2.LineXY] = []
        for eg in edge_groups:
            edge_region = canny[int(eg.top) : int(eg.bottom), int(eg.left) : int(eg.right)]
            edge_idxs = np.where(edge_region == 255)
            # get the correct x/y offsets
            edge_idxs = (edge_idxs[0] + int(eg.top), edge_idxs[1] + int(eg.left))
            # Pxy expects x/y value, but images/np.ndarrays are y-major and so
            # edge_idxs are in y/x order. Flip to x/y order.
            edge_idxs = (edge_idxs[1], edge_idxs[0])
            points = p2.Pxy(edge_idxs)
            line = l2.LineXY.fit_from_points(points)
            lines.append(line)

        # Assign edges to be top, bottom, left, or right
        for line in lines:
            # Lines with slope == 1 are at 45 degrees. Use this halfway point
            # between vertical/horizontal to determine the whether a line should
            # be classified as top/bottom or left/right.
            if np.abs(line.slope) > 1:
                # vertical(ish) line
                if edges["left"] is None:
                    edges["left"] = line
                else:
                    line_x_intercept = line.x_from_y(height / 2)
                    left_x_intercept = edges["left"].x_from_y(height / 2)
                    if line_x_intercept < left_x_intercept:
                        edges["right"] = edges["left"]
                        edges["left"] = line
                    else:
                        edges["right"] = line
            else:
                # horizontal(ish) line
                if edges["top"] is None:
                    edges["top"] = line
                else:
                    line_y_intercept = line.y_from_x(width / 2)
                    left_y_intercept = edges["top"].y_from_x(width / 2)
                    if line_y_intercept < left_y_intercept:
                        edges["bottom"] = edges["top"]
                        edges["top"] = line
                    else:
                        edges["bottom"] = line

        # Assign corners from the edges
        corners["tl"] = p2.Pxy(edges["top"].intersect_with(edges["left"]))
        corners["tr"] = p2.Pxy(edges["top"].intersect_with(edges["right"]))
        corners["br"] = p2.Pxy(edges["bottom"].intersect_with(edges["right"]))
        corners["bl"] = p2.Pxy(edges["bottom"].intersect_with(edges["left"]))

        # build a region from the corners
        vertices = p2.Pxy.from_list(corners.values())
        region = reg2.RegionXY.from_vertices(vertices)

        edge_corners_image = self.visualize_edges_corners(canny, edges, corners)
        self.algimages_assign_edges_and_corners = {
            "assigned_edges_and_corners": edge_corners_image,
            "Encompassing region": self.visualize_region(canny, region),
        }
        self.summary_visualizations.append(("Edges & Corners", edge_corners_image))
        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=2, slide_control=self.slide_control)
            slide.set_title("step6_assign_edges_and_corners")
            for caption, image in self.algimages_assign_edges_and_corners.items():
                slide.add_image(pps.PowerpointImage(image.astype(np.uint8), caption=caption))
            self.ppt_deck.add_slide(slide)
        if debug_edge_assignment:
            self.draw_images(self.algimages_assign_edges_and_corners)

        return edges, corners, region

    def step6_assign_edges_and_corners_polygon(
        self, edge_groups: list[r2.RectXY], canny: np.ndarray, user_points: p2.Pxy, debug_edge_assignment: bool = False
    ) -> tuple[dict[str, p2.Pxy], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Assign edges and corners for arbitrary polygons using the existing edge detection
        and the user-defined points as guidance.

        Parameters
        ----------
        edge_groups : list[r2.RectXY]
            List of edge group regions from step5.
        canny : np.ndarray
            The edges image from the canny algorithm.
        user_points : p2.Pxy
            The user-defined corner points.
        debug_edge_assignment : bool, optional
            True to draw debug images, by default False.

        Returns
        -------
        tuple[dict[str, p2.Pxy], dict[str, p2.Pxy], reg2.RegionXY]
            A tuple containing:
            - Dictionary of edge lines
            - Dictionary of corner points
            - The resulting RegionXY object
        """
        edges = {}
        corners = {}
        region_edges = []

        # Process each segment between consecutive user points
        for i in range(len(user_points.x)):
            pt1 = (int(user_points.x[i]), int(user_points.y[i]))
            pt2 = (int(user_points.x[(i + 1) % len(user_points.x)]), int(user_points.y[(i + 1) % len(user_points.y)]))

            # Find the best edge between these points
            edge_line = self._find_edge_between_points_with_guidance(canny, pt1, pt2, edge_groups)

            if edge_line is not None:
                edge_key = f"edge_{i}"
                edges[edge_key] = edge_line
                region_edges.append(edge_line)

                # Store the corner points
                corner_key = f"corner_{i}"
                corners[corner_key] = p2.Pxy([pt1[0], pt1[1]])

        # build a region from the corners
        vertices = p2.Pxy.from_list(corners.values())
        region = reg2.RegionXY.from_vertices(vertices)

        # Visualize the results
        edge_corners_image = self.visualize_polygon_edges_corners(canny, edges, corners, 0)
        self.algimages_assign_edges_and_corners = {
            "assigned_edges_and_corners": edge_corners_image,
            "Encompassing region": self.visualize_region(canny, region),
        }
        self.summary_visualizations.append(("Edges & Corners", edge_corners_image))

        if self.generate_powerpoint:
            slide = pps.PowerpointSlide.template_content_grid(nrows=1, ncols=2, slide_control=self.slide_control)
            slide.set_title("step6_assign_edges_and_corners_polygon")
            for caption, image in self.algimages_assign_edges_and_corners.items():
                slide.add_image(pps.PowerpointImage(image.astype(np.uint8), caption=caption))
            self.ppt_deck.add_slide(slide)

        if debug_edge_assignment:
            self.draw_images(self.algimages_assign_edges_and_corners)

        return edges, corners, region

    def visualize_edges_corners(
        self, base_image: np.ndarray, edges: dict[str, l2.LineXY], corners: dict[str, p2.Pxy], thickness=2
    ):
        """
        Visualize edges and corners on a base image.

        Parameters
        ----------
        base_image: np.ndarray
            The base image on which to visualize edges and corners.
        edges: dict[str, l2.LineXY]
            A dictionary of edge labels ("top"/"bottom"/"left"/"right") to edge line objects.
        corners: dict[str, p2.Pxy]
            A dictionary of corner labels ("tl"/"tr"/"bl"/"br") to corner point objects.
        thickness: int, optional
            The thickness of the lines and circles to draw. Default is 2.

        Returns
        -------
        vis_image: np.ndarray
            The image with visualized edges and corners.
        """
        vis_image = ir.false_color_reshaper(base_image)
        width, height = vis_image.shape[1], vis_image.shape[0]

        # Visualize the edges
        for edge_label in edges:
            edge: l2.LineXY = edges[edge_label]
            if edge_label in ["left", "right"]:
                pt1 = (int(edge.x_from_y(0)), 0)
                pt2 = (int(edge.x_from_y(height)), height)
            elif edge_label in ["top", "bottom"]:
                pt1 = (0, int(edge.y_from_x(0)))
                pt2 = (width, int(edge.y_from_x(width)))
            else:
                raise RuntimeError
            edge_color = self.edge_colors[edge_label].rgb_255()
            vis_image = cv.line(vis_image, pt1, pt2, edge_color, thickness=thickness, lineType=cv.LINE_AA)

        # Visualize the corners
        for corner_label in corners:
            corner: p2.Pxy = corners[corner_label]
            center = (int(corner.x[0]), int(corner.y[0]))
            corner_color = self.corner_colors[corner_label].rgb_255()
            vis_image = cv.circle(vis_image, center, 7, corner_color, thickness=thickness)

        return vis_image

    def visualize_polygon_edges_corners(
        self,
        base_image: np.ndarray,
        edges: dict[str, l2.LineXY],
        corners: dict[str, p2.Pxy],
        region_index: int = 0,
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Visualize edges and corners for arbitrary polygons with color cycling.

        Parameters
        ----------
        base_image : np.ndarray
            The base image on which to visualize edges and corners.
        edges : dict[str, l2.LineXY]
            A dictionary of edge labels to edge line objects.
        corners : dict[str, p2.Pxy]
            A dictionary of corner labels to corner point objects.
        region_index : int
            Index of the region for color selection.
        thickness : int, optional
            The thickness of the lines and circles to draw. Default is 2.

        Returns
        -------
        np.ndarray
            The image with visualized edges and corners.
        """
        vis_image = ir.false_color_reshaper(base_image)
        width, height = vis_image.shape[1], vis_image.shape[0]

        # Get color for this region
        color = self.region_colors[region_index % len(self.region_colors)]
        edge_color = (int(color[0]), int(color[1]), int(color[2]))

        # Visualize the edges - for arbitrary polygons, we need to handle each edge segment
        for edge_key, edge in edges.items():
            if edge is None:
                continue

            # For arbitrary polygons, we'll draw the edge between its corner points
            # Extract corner indices from edge key (assuming format "edge_N")
            try:
                edge_idx = int(edge_key.replace('edge_', ''))
                corner1_key = f"corner_{edge_idx}"
                corner2_key = f"corner_{(edge_idx + 1) % len(corners)}"

                if corner1_key in corners and corner2_key in corners:
                    pt1 = (int(corners[corner1_key].x[0]), int(corners[corner1_key].y[0]))
                    pt2 = (int(corners[corner2_key].x[0]), int(corners[corner2_key].y[0]))
                    cv.line(vis_image, pt1, pt2, edge_color, thickness, lineType=cv.LINE_AA)
            except (ValueError, KeyError):
                # Fallback: draw the edge line across the image
                if abs(edge.slope) > 1:  # More vertical
                    pt1 = (int(edge.x_from_y(0)), 0)
                    pt2 = (int(edge.x_from_y(height - 1)), height - 1)
                else:  # More horizontal
                    pt1 = (0, int(edge.y_from_x(0)))
                    pt2 = (width - 1, int(edge.y_from_x(width - 1)))
                cv.line(vis_image, pt1, pt2, edge_color, thickness, lineType=cv.LINE_AA)

        # Visualize the corners with cycling colors
        for i, (corner_key, corner) in enumerate(corners.items()):
            if corner is None:
                continue

            # Cycle through colors for corners
            corner_color_idx = (region_index + i) % len(self.region_colors)
            corner_color = self.region_colors[corner_color_idx]
            corner_color = (int(corner_color[0]), int(corner_color[1]), int(corner_color[2]))

            center = (int(corner.x[0]), int(corner.y[0]))
            cv.circle(vis_image, center, 7, corner_color, thickness)

            # Add corner label
            label = corner_key.replace('corner_', '')
            text_pos = (center[0] + 10, center[1] - 5)
            cv.putText(vis_image, label, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv.LINE_AA)

        return vis_image

    def visualize_region(self, base_image: np.ndarray, region: reg2.RegionXY) -> np.ndarray:
        """
        Visualize a region on a base image. Sets all pixels to 255 within the
        given region.

        Parameters
        ----------
        base_image: np.ndarray
            The base image on which to visualize the region.
        region: reg2.RegionXY
            The region to visualize.

        Returns
        -------
        vis_image: np.ndarray
            The image with the visualized region.
        """
        vis_image = base_image.copy()
        width, height = base_image.shape[1], base_image.shape[0]

        # Build the pixels index list
        all_x = list(range(width)) * height
        all_y = np.zeros((width * height))
        for y in range(height):
            all_y[y * width : (y + 1) * width] = y
        all_xy = p2.Pxy([all_x, all_y])

        # Visualize the region
        matching_xy = all_xy[np.where(region.is_inside(all_xy))]
        vis_image[matching_xy.asindex("yx")] = 255

        return vis_image

    def find_boundary_pixels_in_image(
        self,
        image: np.ndarray,
        approx_center_pixel: p2.Pxy,
        debug_canny_settings=False,
        debug_blob_analysis=False,
        debug_ray_projection=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find the pixels that are good candidates for the boundaries of a the
        region within the given image.

        Parameters
        ----------
        image: np.ndarray
            The input image.
        approx_center_pixel: p2.Pxy
            The approximate center pixel.
        debug_canny_settings: bool, optional
            Flag to enable debugging for Canny edge detection. Default is False.
        debug_blob_analysis: bool, optional
            Flag to enable debugging for blob analysis. Default is False.
        debug_ray_projection: bool, optional
            Flag to enable debugging for ray projection. Default is False.

        Returns
        -------
        canny_edges_boundary_pixels: tuple[np.ndarray, np.ndarray]
            A tuple containing numpy arrays, with the Canny edge pixels from the
            Canny edge detector and the resulting boundary edge pixels from
            steps 1-4.
        """
        # Reset the debug images
        self.ppt_deck = rcpp.RenderControlPowerpointPresentation()
        self.images_to_show.clear()
        self.summary_visualizations.clear()

        # Apply canny edge detection
        canny_edges = self.step1_canny_edge_detection(debug_canny_settings, image)
        # Image.fromarray(canny_edges).save(orp.opencsp_temporary_dir() + "/tmp/canny_edges.png")
        # canny_edges = np.array(Image.open(orp.opencsp_temporary_dir() + "/tmp/canny_edges.png"))

        # Use blobs to create a "wall" that can be easiliy detected.
        horizontal_mask, vertical_mask, negative_mask = self.step2_blob_analysis(canny_edges, debug_blob_analysis)
        thin_boundaries = np.ones_like(canny_edges, dtype=np.uint8)
        thin_boundaries[negative_mask] = 0
        # Image.fromarray(thin_boundaries).save(orp.opencsp_temporary_dir() + "/tmp/thin_boundaries.png")
        # thin_boundaries = np.array(Image.open(orp.opencsp_temporary_dir() + "/tmp/thin_boundaries.png"))

        # Use rays to find the boundaries of the wall from within the target area
        rectangle_center, boundary_samples = self.step3_ray_project_edge_intercept(
            thin_boundaries, approx_center_pixel, debug_ray_projection
        )
        # np.save(orp.opencsp_temporary_dir() + "/tmp/rectangle_center", rectangle_center.data)
        # np.save(orp.opencsp_temporary_dir() + "/tmp/boundary_samples", boundary_samples.data)
        # rectangle_center = p2.Pxy(np.load(orp.opencsp_temporary_dir() + "/tmp/rectangle_center.npy"))
        # boundary_samples = p2.Pxy(np.load(orp.opencsp_temporary_dir() + "/tmp/boundary_samples.npy"))

        # Get the edge pixels from the boundary locations
        boundary_edges = self.step4_boundary_locations_to_edges(
            boundary_samples, canny_edges, debug_ray_projection=debug_ray_projection
        )

        return canny_edges, boundary_edges

    def find_rectangular_region(
        self, boundary_edges: p2.Pxy, canny_edges: np.ndarray, debug_edge_groups=False, debug_edge_assignment=False
    ) -> tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Find a rectangular region from the given boundary and canny edge images.

        Parameters
        ----------
        boundary_edges: p2.Pxy
            The boundary edges, as a numpy array with zeros for non-matching
            pixels and non-zeros for potential boundary pixels.
        canny_edges: np.ndarray
            The Canny edges, as a numpy array.
        debug_edge_groups: bool, optional
            Flag to enable debugging for edge groups. Default is False.
        debug_edge_assignment: bool, optional
            Flag to enable debugging for edge groups assignment. Default is False.

        Returns
        -------
        edges: dict[str, l2.LineXY]
            The edges of the rectangular region ("top", "bottom", "left", and "right").
        corners: dict[str, p2.Pxy]
            The corners of the rectangular region ("tl", "tr", "bl", and "br").
        region: reg2.RegionXY
            The located rectangular region.
        """
        # Scan for the left, right, top, and bottom edge groups
        edge_groups = self.step5_find_edge_groups(boundary_edges, debug_edge_groups)

        # Find the target region from the edge groups
        edges, corners, region = self.step6_assign_edges_and_corners(edge_groups, canny_edges, debug_edge_assignment)

        return edges, corners, region

    def save_debug_images(self, powerpoint_save_file: str, overwrite=False):
        """
        Saves the debug images to a powerpoint file. This is useful for
        debugging the region detection steps.

        Parameters
        ----------
        powerpoint_save_file: str
            The path/name.ext to save to.
        overwrite: bool, optional
            Flag to allow overwriting the file if it exists. Default is False.

        Raises
        ------
        FileNotFoundError:
            If the path to save to does not exist.
        FileExistsError:
            If the PowerPoint file already exists and overwrite is not allowed.
        """
        # Validate input
        ppt_path, ppt_name, ppt_ext = ft.path_components(powerpoint_save_file)
        if not ft.directory_exists(ppt_path):
            lt.error_and_raise(
                FileNotFoundError,
                "Error in RegionDetector.find_region(): "
                + f"can't find the directory {ppt_path} to save the powerpoint to",
            )
        if ft.file_exists(powerpoint_save_file):
            if not overwrite:
                lt.error_and_raise(
                    FileExistsError,
                    "Error in RegionDetector.find_region(): " + f"file {powerpoint_save_file} already exists",
                )

        # Save the visualizations to a powerpoint
        if powerpoint_save_file is not None:
            self.ppt_deck.save(powerpoint_save_file, overwrite)

    def find_region_from_interactive_points(
        self, image: np.ndarray, debug: bool = False, output_powerpoint: str = None
    ) -> tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]:
        """
        Interactively select points in an image and find the best edges connecting them to form a polygon region.

        This method allows the user to manually select key points in the image, then uses the existing
        edge detection pipeline to find the optimal edges connecting these points to form a closed region.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        debug : bool, optional
            If True, shows intermediate debug images (default: False).
        output_powerpoint : str, optional
            Path to save debug PowerPoint presentation (default: None).

        Returns
        -------
        tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]
            A tuple containing:
            - Dictionary of edge lines connecting the points
            - Dictionary of corner points (both user-selected and computed intersections)
            - The resulting RegionXY object

        Key Bindings:
        - Left Click: Add a new point
        - Right Click: Remove the most recent point
        - 'c': Close the polygon by connecting to the first point
        - ENTER/SPACE: Finish selection and process the region
        - ESC: Cancel and restart the selection
        """
        # Reset debug images
        self.ppt_deck = rcpp.RenderControlPowerpointPresentation()
        self.images_to_show.clear()
        self.summary_visualizations.clear()

        # Convert to color if needed for visualization
        display_image = image.copy()
        if len(display_image.shape) == 2 or display_image.shape[2] == 1:
            display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)

        # Create window for interactive point selection
        window_name = "Select Region Points"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # Create instructions window
        instructions = (
            "INSTRUCTIONS:\n\n"
            "Left Click: Add a new point\n"
            "Right Click: Remove the most recent point\n"
            "c: Close polygon (connect to first point)\n"
            "ENTER/SPACE: Finish selection\n"
            "ESC: Cancel and restart\n\n"
            "Click on key points to define the\n"
            "region boundaries.\n"
            "Current implementation expects\n"
            "a rectanglular region\n"
            "starting in the top right\n"
            "corner proceeding counterclockwise."
        )

        # Show instructions in a popup that stays visible
        cv.namedWindow("Instructions", cv.WINDOW_NORMAL)
        instructions_img = np.zeros((400, 600, 3), dtype=np.uint8)
        instructions_img.fill(240)  # Light gray background

        # Put text on the instructions image
        y0, dy = 30, 25
        for i, line in enumerate(instructions.split('\n')):
            y = y0 + i * dy
            cv.putText(instructions_img, line, (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("Instructions", instructions_img)

        # Initial display
        cv.imshow(window_name, display_image)

        # Mouse callback function
        selected_points = []
        polygon_closed = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_points, display_image, polygon_closed

            if event == cv.EVENT_LBUTTONDOWN and not polygon_closed:
                # Add point to list
                selected_points.append((x, y))

                # Draw point on image
                cv.circle(display_image, (x, y), 5, (0, 0, 255), -1)

                # Draw connecting lines if we have multiple points
                if len(selected_points) > 1:
                    cv.line(display_image, selected_points[-2], selected_points[-1], (0, 255, 0), 2)

                # Show updated image
                cv.imshow(window_name, display_image)

            elif event == cv.EVENT_RBUTTONDOWN and not polygon_closed:
                # Remove the most recent point if we have any
                if selected_points:
                    selected_points.pop()

                    # Redraw the image without the last point
                    display_image = image.copy()
                    if len(display_image.shape) == 2 or display_image.shape[2] == 1:
                        display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)

                    # Redraw remaining points and connections
                    for i in range(len(selected_points)):
                        cv.circle(display_image, selected_points[i], 5, (0, 0, 255), -1)
                        if i > 0:
                            cv.line(display_image, selected_points[i - 1], selected_points[i], (0, 255, 0), 2)

                    cv.imshow(window_name, display_image)

        # Set mouse callback
        cv.setMouseCallback(window_name, mouse_callback)

        print("Interactive region selection started. See Instructions window for help.")

        # Wait for user to finish selection
        while True:
            key = cv.waitKey(1) & 0xFF

            if key == ord('c') and len(selected_points) > 2 and not polygon_closed:
                # Close the polygon by connecting to the first point
                if len(selected_points) > 1:
                    cv.line(display_image, selected_points[-1], selected_points[0], (0, 0, 255), 2)
                    cv.imshow(window_name, display_image)
                    polygon_closed = True
                    print("Polygon closed. Press ENTER/SPACE to process or continue adding points.")

            elif key in [13, 32] and len(selected_points) >= 3:  # ENTER or SPACE
                break

            elif key == 27:  # ESC
                selected_points = []
                polygon_closed = False
                display_image = image.copy()
                if len(display_image.shape) == 2 or display_image.shape[2] == 1:
                    display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)
                cv.imshow(window_name, display_image)
                print("Selection cancelled. Start over.")

        # Clean up instruction window
        cv.destroyWindow("Instructions")

        if len(selected_points) < 3:
            cv.destroyWindow(window_name)
            raise ValueError("At least 3 points are required to define a region.")

        # If polygon wasn't explicitly closed, close it automatically
        if not polygon_closed and len(selected_points) > 1:
            cv.line(display_image, selected_points[-1], selected_points[0], (0, 0, 255), 2)
            cv.imshow(window_name, display_image)

        # Close the selection window
        cv.destroyWindow(window_name)

        # Convert selected points to Pxy format
        points_x = [p[0] for p in selected_points]
        points_y = [p[1] for p in selected_points]
        user_points = p2.Pxy([points_x, points_y])

        # Run Canny edge detection on the original image
        canny_edges = self.step1_canny_edge_detection(debug, image)

        # Find edges between consecutive points (including closing the polygon)
        edges = {}
        corners = {}
        region_edges = []
        edge_lable_list = ["top", "left", "bottom", "right"]
        corner_lable_list = ["tr", "tl", "bl", "br"]

        # Process each segment between consecutive points
        for i in range(len(selected_points)):
            pt1 = selected_points[i]
            pt2 = selected_points[(i + 1) % len(selected_points)]  # Wrap around for closing the polygon

            # Create a region of interest around this segment
            segment_image = self._create_segment_roi(canny_edges, pt1, pt2)

            # Find the best edge line between these points
            edge_line = self._find_best_edge_between_points(segment_image, pt1, pt2)

            if edge_line is not None:
                try:
                    edge_key = edge_lable_list[i]
                except:
                    edge_key = f"edge_{i}"
                edges[edge_key] = edge_line
                region_edges.append(edge_line)

                # Store the endpoints as corners
                try:
                    corner_key = corner_lable_list[i]
                    corners[corner_key] = p2.Pxy([pt1[0], pt1[1]])
                except:
                    corner_key = f"corner_{i}"
                    corners[corner_key] = p2.Pxy([pt1[0], pt1[1]])

        # Create the region from the collected edges
        region = self._create_region_from_edges(region_edges, user_points)

        # Visualize the results
        result_image = self.visualize_edges_corners(canny_edges, edges, corners)
        self.summary_visualizations.append(("Final Region", result_image))

        # Save PowerPoint if requested
        if output_powerpoint is not None:
            self.save_debug_images(output_powerpoint, overwrite=True)

        if debug:
            self.draw_images(self.summary_visualizations)

        return edges, corners, region

    def find_multiple_regions_from_interactive_points(
        self, image: np.ndarray, debug: bool = False, output_powerpoint: str = None, output_data_dir: str = None
    ) -> dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]:
        """
        Interactively select multiple regions in an image with names and find edges connecting points.

        This enhanced method allows selecting multiple regions in a single image, naming each region
        through a GUI dialog, and supports arbitrary polygon shapes. Regions should be defined
        counter-clockwise starting from any point. When regions overlap, shared edges are assigned
        to the oldest region and corner points are synchronized.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.
        debug : bool, optional
            If True, shows intermediate debug images (default: False).
        output_powerpoint : str, optional
            Path to save debug PowerPoint presentation (default: None).
        output_data_dir : str, optional
            Path to directory to save created regions and images for reloading (default: None)

        Returns
        -------
        dict[str, tuple[dict[str, l2.LineXY], dict[str, p2.Pxy], reg2.RegionXY]]
            A dictionary where keys are region names and values are tuples containing:
            - Dictionary of edge lines for the region
            - Dictionary of corner points
            - The resulting RegionXY object

        Key Bindings:
        - Left Click: Add a new point to current region
        - Right Click: Remove the most recent point
        - 'c': Close current polygon (connect to first point)
        - 'n': Name current region via GUI dialog and start new region
        - ENTER/SPACE: Finish all region selection
        - ESC: Cancel current region and restart
        """
        # Import tkinter for GUI dialogs
        try:
            import tkinter as tk
            from tkinter import simpledialog
        except ImportError:
            raise ImportError("tkinter is required for this function. Please install it.")

        # Reset debug images
        self.ppt_deck = rcpp.RenderControlPowerpointPresentation()
        self.images_to_show.clear()
        self.summary_visualizations.clear()

        # Convert to color if needed for visualization
        display_image = image.copy()
        if len(display_image.shape) == 2 or display_image.shape[2] == 1:
            display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)

        # Create window for interactive point selection
        window_name = "Select Multiple Regions"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # Create instructions window
        instructions = (
            "INSTRUCTIONS - Multiple Region Selection:\n\n"
            "Current Region Controls:\n"
            "  Left Click: Add point to current region\n"
            "  Right Click: Remove last point\n"
            "  c: Close current polygon\n"
            "  n: Name current region (GUI dialog)\n"
            "  m: Toggle mask mode\n"
            "  f: Finalize current mask\n"
            "  d: Delete current mask\n"
            "Snap Controls:\n"
            "  s: Toggle snap to points (ON/OFF)\n"
            "  r: Toggle snap mask to region points (ON/OFF)\n"
            "Global Controls:\n"
            "  ENTER/SPACE: Finish all regions\n"
            "  ESC: Cancel current region\n\n"
            "Define regions counter-clockwise.\n"
            "Overlapping edges will be shared."
        )

        # Show instructions in a popup that stays visible
        cv.namedWindow("Instructions", cv.WINDOW_NORMAL)
        instructions_img = np.zeros((800, 500, 3), dtype=np.uint8)
        instructions_img.fill(240)  # Light gray background

        # Put text on the instructions image
        y0, dy = 30, 25
        for i, line in enumerate(instructions.split('\n')):
            y = y0 + i * dy
            cv.putText(instructions_img, line, (20, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.imshow("Instructions", instructions_img)

        # Initial display
        cv.imshow(window_name, display_image)

        # State variables
        all_regions_data = []  # List to store (name, points) tuples before processing
        current_points = []  # Points for current region being defined
        current_polygon_closed = False
        region_counter = 1
        mouse_position = (0, 0)  # Track mouse position for crosshairs
        snap_to_points = True  # Enable snapping to existing points
        snap_to_region_points = True  # Enable snapping mask points to region points
        mask_mode = False
        mask_points = []
        mask_polygon_closed = False
        masks = []  # List to store completed masks

        # Color palette for different regions
        self.region_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
        ]

        def get_region_color(region_index):
            """Get color for a region based on its index"""
            return self.region_colors[region_index % len(self.region_colors)]

        def redraw_display():
            """Redraw the display image with all regions, current points, and masks"""
            nonlocal display_image

            # Start with original image
            display_image = image.copy()
            if len(display_image.shape) == 2 or display_image.shape[2] == 1:
                display_image = cv.cvtColor(display_image, cv.COLOR_GRAY2BGR)

            # Apply masks (darken masked areas)
            for mask in masks:
                mask_color = (50, 50, 50)  # Dark gray for masked areas
                cv.fillPoly(display_image, [np.array(mask, np.int32)], mask_color)

            # Draw all completed regions
            for i, (name, points) in enumerate(all_regions_data):
                color = get_region_color(i)

                # Draw points
                for pt in points:
                    cv.circle(display_image, pt, 5, color, -1)

                # Draw connecting lines
                for j in range(len(points) - 1):
                    cv.line(display_image, points[j], points[j + 1], color, 2)

                # Draw closing line
                if len(points) > 1:
                    cv.line(display_image, points[-1], points[0], color, 2)

                # Draw region name near first point
                if points:
                    text_pos = (points[0][0] + 10, points[0][1] - 10)
                    cv.putText(display_image, name, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

            # Draw current region being defined
            if current_points:
                color = get_region_color(len(all_regions_data))

                # Draw points
                for pt in current_points:
                    cv.circle(display_image, pt, 5, color, -1)

                # Draw connecting lines
                for i in range(len(current_points) - 1):
                    cv.line(display_image, current_points[i], current_points[i + 1], color, 2)

                # Draw closing line if polygon is closed
                if current_polygon_closed and len(current_points) > 1:
                    cv.line(display_image, current_points[-1], current_points[0], color, 2)

            # Draw mask being defined
            if mask_mode and mask_points:
                mask_color = (0, 0, 255)  # Red for mask points

                # Draw points
                for pt in mask_points:
                    cv.circle(display_image, pt, 5, mask_color, -1)

                # Draw connecting lines
                for i in range(len(mask_points) - 1):
                    cv.line(display_image, mask_points[i], mask_points[i + 1], mask_color, 2)

                # Draw closing line if polygon is closed
                if mask_polygon_closed and len(mask_points) > 1:
                    cv.line(display_image, mask_points[-1], mask_points[0], mask_color, 2)

            # Draw dynamic crosshairs at mouse position
            cv.line(
                display_image, (mouse_position[0], 0), (mouse_position[0], display_image.shape[0]), (255, 255, 255), 1
            )
            cv.line(
                display_image, (0, mouse_position[1]), (display_image.shape[1], mouse_position[1]), (255, 255, 255), 1
            )

            if snap_to_points or (mask_mode and snap_to_region_points):
                # Draw snap indicators if near existing points
                snap_radius = 10
                for i, (name, points) in enumerate(all_regions_data):
                    for pt in points:
                        if (
                            abs(pt[0] - mouse_position[0]) < snap_radius
                            and abs(pt[1] - mouse_position[1]) < snap_radius
                        ):
                            cv.circle(display_image, pt, snap_radius, (0, 255, 255), 1)

                if current_points:
                    for pt in current_points:
                        if (
                            abs(pt[0] - mouse_position[0]) < snap_radius
                            and abs(pt[1] - mouse_position[1]) < snap_radius
                        ):
                            cv.circle(display_image, pt, snap_radius, (0, 255, 255), 1)

            # Draw snap status indicators
            snap_status = []
            if snap_to_points:
                snap_status.append("Snap: ON")
            else:
                snap_status.append("Snap: OFF")

            if mask_mode and snap_to_region_points:
                snap_status.append("Mask Snap: ON")
            elif mask_mode:
                snap_status.append("Mask Snap: OFF")

            if snap_status:
                status_text = " | ".join(snap_status)
                cv.putText(
                    display_image, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA
                )

            cv.imshow(window_name, display_image)

        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points, current_polygon_closed, mouse_position
            nonlocal mask_points, mask_polygon_closed
            nonlocal snap_to_points, snap_to_region_points

            mouse_position = (x, y)

            if mask_mode:
                if event == cv.EVENT_LBUTTONDOWN and not mask_polygon_closed:
                    # Add point to mask
                    snapped_x, snapped_y = x, y

                    # Check if we should snap to region points
                    if snap_to_region_points and all_regions_data:
                        snap_radius = 10
                        for name, points in all_regions_data:
                            for pt in points:
                                if abs(pt[0] - x) < snap_radius and abs(pt[1] - y) < snap_radius:
                                    snapped_x, snapped_y = pt
                                    break
                            if (snapped_x, snapped_y) != (x, y):
                                break

                    mask_points.append((snapped_x, snapped_y))
                    redraw_display()

                elif event == cv.EVENT_RBUTTONDOWN and not mask_polygon_closed:
                    # Remove the most recent point if we have any
                    if mask_points:
                        mask_points.pop()
                        redraw_display()

                elif event == cv.EVENT_MOUSEMOVE:
                    # Update mouse position for crosshairs
                    redraw_display()

            else:
                if event == cv.EVENT_LBUTTONDOWN and not current_polygon_closed:
                    # Check if we should snap to existing points
                    if snap_to_points:
                        snap_radius = 10
                        snapped = False

                        # Check against all existing points
                        for name, points in all_regions_data:
                            for pt in points:
                                if abs(pt[0] - x) < snap_radius and abs(pt[1] - y) < snap_radius:
                                    x, y = pt
                                    snapped = True
                                    break
                            if snapped:
                                break

                        if not snapped and current_points:
                            for pt in current_points:
                                if abs(pt[0] - x) < snap_radius and abs(pt[1] - y) < snap_radius:
                                    x, y = pt
                                    snapped = True
                                    break

                    # Add point to current region
                    current_points.append((x, y))
                    redraw_display()

                elif event == cv.EVENT_RBUTTONDOWN and not current_polygon_closed:
                    # Remove the most recent point if we have any
                    if current_points:
                        current_points.pop()
                        redraw_display()

                elif event == cv.EVENT_MOUSEMOVE:
                    # Update mouse position for crosshairs
                    redraw_display()

        # Set mouse callback
        cv.setMouseCallback(window_name, mouse_callback)

        print("Interactive multi-region selection started. See Instructions window for help.")

        # Update the main interaction loop to add toggle for snap_to_region_points:
        while True:
            key = cv.waitKey(1) & 0xFF

            if key == ord('c') and len(current_points) > 2 and not current_polygon_closed:
                # Close the current polygon
                current_polygon_closed = True
                print(f"Polygon closed. Press 'n' to name this region and start a new one.")
                redraw_display()

            elif key == ord('n') and current_points and (current_polygon_closed or len(current_points) > 2):
                # Name the current region via GUI dialog
                root = tk.Tk()
                root.withdraw()  # Hide the main window

                default_name = f"Region_{region_counter}"
                region_name = simpledialog.askstring(
                    "Region Name", f"Enter name for current region:", initialvalue=default_name, parent=root
                )

                if region_name is None:  # User cancelled
                    root.destroy()
                    continue

                region_name = region_name.strip()
                if not region_name:
                    region_name = default_name

                # Check if name already exists
                if any(name == region_name for name, _ in all_regions_data):
                    response = simpledialog.askstring(
                        "Name Conflict",
                        f"Region '{region_name}' already exists. Enter a different name:",
                        initialvalue=f"{region_name}_{region_counter}",
                        parent=root,
                    )
                    if response is None:
                        root.destroy()
                        continue
                    region_name = response.strip()
                    if not region_name:
                        region_name = f"{default_name}_{region_counter}"

                root.destroy()

                # Store the region data (name and points) for later processing
                all_regions_data.append((region_name, current_points.copy()))
                print(f"Region '{region_name}' saved with {len(current_points)} points.")

                # Reset for next region
                current_points = []
                current_polygon_closed = False
                region_counter += 1

                redraw_display()

            elif key == ord('m'):
                # Toggle mask mode
                mask_mode = not mask_mode
                if mask_mode:
                    print("Mask mode activated. Define mask regions to exclude from analysis.")
                else:
                    print("Mask mode deactivated. Back to region selection.")
                redraw_display()

            elif key == ord('f') and mask_mode and mask_points and (mask_polygon_closed or len(mask_points) > 2):
                # Finalize current mask
                if not mask_polygon_closed and len(mask_points) > 1:
                    # Close the mask polygon if not already closed
                    mask_points.append(mask_points[0])
                    mask_polygon_closed = True
                    redraw_display()

                # Add mask to the list of masks
                masks.append(mask_points.copy())
                print(f"Mask saved with {len(mask_points)} points.")

                # Reset mask points
                mask_points = []
                mask_polygon_closed = False
                redraw_display()

            elif key == ord('d') and mask_mode and mask_points:
                # Delete current mask being defined
                mask_points = []
                mask_polygon_closed = False
                print("Current mask deleted.")
                redraw_display()

            elif key == ord('s'):
                # Toggle snap to points
                snap_to_points = not snap_to_points
                if snap_to_points:
                    print("Snap to points: ON")
                else:
                    print("Snap to points: OFF")
                redraw_display()

            elif key == ord('r'):
                # Toggle snap mask points to region points
                snap_to_region_points = not snap_to_region_points
                if snap_to_region_points:
                    print("Snap mask points to region points: ON")
                else:
                    print("Snap mask points to region points: OFF")
                redraw_display()

            elif key in [13, 32] and len(all_regions_data) > 0:  # ENTER or SPACE
                # Finish all region selection
                break

            elif key == 27:  # ESC
                # Cancel current region or mask
                if mask_mode and mask_points:
                    print("Current mask cancelled.")
                    mask_points = []
                    mask_polygon_closed = False
                elif current_points:
                    print("Current region cancelled.")
                    current_points = []
                    current_polygon_closed = False
                else:
                    print("No current region or mask to cancel.")
                redraw_display()

        # Clean up instruction window
        cv.destroyWindow("Instructions")
        cv.destroyWindow(window_name)

        if not all_regions_data:
            raise ValueError("No regions were defined.")

        print(f"Processing {len(all_regions_data)} regions...")

        # Process all regions at once
        all_regions = {}
        # Create a mask image
        mask_image = np.ones_like(image, dtype=np.uint8) * 255  # Start with all white (no mask)

        # Apply all masks to the mask image
        for mask in masks:
            mask_array = np.array(mask, np.int32)
            cv.fillPoly(mask_image, [mask_array], 0)  # Fill mask areas with black (0)

        # Convert mask to single channel if needed
        if len(mask_image.shape) == 3:
            mask_image = cv.cvtColor(mask_image, cv.COLOR_BGR2GRAY)

        # Apply the mask to the original image
        masked_image = cv.bitwise_and(image, image, mask=mask_image)

        # Use the masked image for edge detection
        # canny_edges = self.step1_canny_edge_detection(debug, masked_image)
        processed_regions = []  # Store processed regions for overlap checking

        for i, (region_name, points) in enumerate(all_regions_data):
            try:
                # Convert points to Pxy format
                points_x = [p[0] for p in points]
                points_y = [p[1] for p in points]
                user_points = p2.Pxy([points_x, points_y])

                # Create temporary region for overlap checking
                temp_region = reg2.RegionXY.from_vertices(user_points)

                # Check for overlaps with existing temporary regions
                if i > 0:
                    for j, (existing_name, existing_points) in enumerate(all_regions_data[:i]):
                        existing_temp_region = reg2.RegionXY.from_vertices(
                            p2.Pxy([existing_points[0], existing_points[1]])
                        )

                        # Get edge samples from both regions
                        edge_samples = temp_region.edge_sample(100)
                        existing_edge_samples = existing_temp_region.edge_sample(100)

                        # Check if any edge points are inside the other region
                        overlap = np.any(existing_temp_region.is_inside_or_on_border(edge_samples))
                        overlap |= np.any(temp_region.is_inside_or_on_border(existing_edge_samples))

                        if overlap:
                            print(f"Found overlap between '{region_name}' and '{existing_name}'")
                            # Modify points to eliminate overlap
                            points = self._resolve_region_overlap(
                                points, existing_points, temp_region, existing_temp_region
                            )
                            user_points = p2.Pxy([points_x, points_y])
                            temp_region = reg2.RegionXY.from_vertices(user_points)
                            print(f"Resolved overlap for region '{region_name}'")

                # Use the existing step functions to find edges
                canny_edges = self.step1_canny_edge_detection(debug, masked_image)
                horizontal_mask, vertical_mask, negative_mask = self.step2_blob_analysis(canny_edges, debug)
                thin_boundaries = np.ones_like(canny_edges, dtype=np.uint8)
                thin_boundaries[negative_mask] = 0

                # Calculate approximate center for ray projection
                approx_center = p2.Pxy([np.mean(points_x), np.mean(points_y)])
                rectangle_center, boundary_samples = self.step3_ray_project_edge_intercept(
                    thin_boundaries, approx_center, debug
                )

                # Get boundary edges
                boundary_edges = self.step4_boundary_locations_to_edges(
                    boundary_samples, canny_edges, debug_ray_projection=debug
                )

                # Find edge groups (this will need modification for arbitrary polygons)
                edge_groups = self.step5_find_edge_groups(boundary_edges, debug)

                # Use new step6 for arbitrary polygons
                edges, corners, region = self.step6_assign_edges_and_corners_polygon(
                    edge_groups, canny_edges, user_points, debug
                )

                # Store the completed region
                region.add_name(region_name)
                all_regions[region_name] = (edges, corners, region)
                processed_regions.append((region_name, (edges, corners, region)))
                print(f"Processed region '{region_name}' successfully.")

            except Exception as e:
                print(f"Error processing region '{region_name}': {str(e)}")
                continue

        if not all_regions:
            raise ValueError("No valid regions were processed.")

        print(f"Completed selection of {len(all_regions)} regions.")

        # Save PowerPoint if requested
        if output_powerpoint is not None:
            self.save_debug_images(output_powerpoint, overwrite=True)

        if debug:
            # Show summary visualizations for each region
            for i, (name, (edges, corners, region)) in enumerate(all_regions.items()):
                print(f"\nShowing debug images for region: {name}")
                result_image = self.visualize_polygon_edges_corners(canny_edges, edges, corners, i, thickness=2)
                self.summary_visualizations.append((f"Region: {name}", result_image))
            self.draw_images(self.summary_visualizations)

        if output_data_dir:
            try:
                cv.imwrite(ft.join(output_data_dir, "interactive_drawn_region_init_with_mask.png"), masked_image)
            except:
                print("Error saving interactive drawn region and mask image")

            try:
                self.export_regions_to_json(all_regions, ft.join(output_data_dir, "region_data.json"))
                # foo = self.import_regions_from_json(ft.join(output_data_dir, "region_data.json"))
            except:
                print("Error saving interactive region data to json file")

            try:
                self.export_masks_to_json(masks, ft.join(output_data_dir, "mask_data.json"))
                # bar = self.import_masks_from_json(ft.join(output_data_dir, "mask_data.json"))
            except:
                print("Error saving interactive mask data to json file")

        return all_regions, masks

    def _find_edge_between_points_with_guidance(
        self, canny: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], edge_groups: list[r2.RectXY]
    ) -> l2.LineXY:
        """
        Find the best edge between two points using the edge groups as guidance.

        Parameters
        ----------
        canny : np.ndarray
            Canny edge detection result.
        pt1 : tuple[int, int]
            First point (x, y).
        pt2 : tuple[int, int]
            Second point (x, y).
        edge_groups : list[r2.RectXY]
            Edge groups from step5.

        Returns
        -------
        l2.LineXY or None
            The best fit line for the edge between the points, or None if no good edge found.
        """
        # Create a region of interest around the line segment
        segment_image = self._create_segment_roi(canny, pt1, pt2)

        # Find edge points in the ROI
        edge_y, edge_x = np.where(segment_image > 0)
        if len(edge_x) < 10:
            # Not enough edge points - return line between points
            return l2.LineXY.from_two_points(p2.Pxy([pt1[0], pt1[1]]), p2.Pxy([pt2[0], pt2[1]]))

        edge_points = p2.Pxy([edge_x, edge_y])

        # Try to fit a line to these points
        try:
            line = l2.LineXY.fit_from_points(edge_points)
            return line
        except:
            # If fitting fails, return line between points
            return l2.LineXY.from_two_points(p2.Pxy([pt1[0], pt1[1]]), p2.Pxy([pt2[0], pt2[1]]))

    def _resolve_region_overlap(
        self,
        points: list[tuple[int, int]],
        existing_points: list[tuple[int, int]],
        temp_region: reg2.RegionXY,
        existing_temp_region: reg2.RegionXY,
    ) -> list[tuple[int, int]]:
        """
        Modify the points of a new region to eliminate overlap with an existing region.

        Parameters
        ----------
        points : list[tuple[int, int]]
            Points of the new region to modify.
        existing_points : list[tuple[int, int]]
            Points of the existing region.
        temp_region : RegionXY
            Temporary region of the new points.
        existing_temp_region : RegionXY
            Temporary region of the existing points.

        Returns
        -------
        list[tuple[int, int]]
            Modified points with no overlap.
        """
        # Convert points to numpy arrays for easier manipulation
        points_array = np.array(points)
        existing_points_array = np.array(existing_points)

        # Find which points of the new region are inside the existing region
        new_region_points = p2.Pxy([points_array[:, 0], points_array[:, 1]])
        is_inside = existing_temp_region.is_inside_or_on_border(new_region_points)

        # If no points are inside, return original points
        if not np.any(is_inside):
            return points

        # Find the closest point on the existing region's boundary for each overlapping point
        modified_points = points.copy()
        for i, point in enumerate(points):
            if is_inside[i]:
                # Find closest point on existing region's boundary
                closest_point = self._find_closest_boundary_point(p2.Pxy([point[0], point[1]]), existing_temp_region)
                modified_points[i] = (int(closest_point.x[0]), int(closest_point.y[0]))

        return modified_points

    def _find_closest_boundary_point(self, point: p2.Pxy, region: reg2.RegionXY, num_samples: int = 1000) -> p2.Pxy:
        """
        Find the closest point on a region's boundary to the given point.

        Parameters
        ----------
        point : Pxy
            The point to find the closest boundary point to.
        region : RegionXY
            The region to sample boundary points from.
        num_samples : int
            Number of boundary points to sample.

        Returns
        -------
        Pxy
            The closest boundary point.
        """
        # Sample boundary points
        boundary_points = region.edge_sample(num_samples)

        # Calculate distances
        deltas = boundary_points - point
        distances = deltas.magnitude()

        # Find the closest point
        min_idx = np.argmin(distances)
        return boundary_points[min_idx]

    def _create_region_from_edges(self, edges: list[l2.LineXY], user_points: p2.Pxy) -> reg2.RegionXY:
        """
        Create a RegionXY object from the collected edges and user points.

        Parameters
        ----------
        edges : list[l2.LineXY]
            List of edge lines.
        user_points : p2.Pxy
            The original user-selected points.

        Returns
        -------
        reg2.RegionXY
            The resulting region object.
        """
        # Collect all intersection points between consecutive edges
        vertices = []

        # Add user points as vertices
        for i in range(len(user_points.x)):
            vertices.append(p2.Pxy([user_points.x[i], user_points.y[i]]))

        # Create the region from vertices
        try:
            region = reg2.RegionXY.from_vertices(p2.Pxy.from_list(vertices))
            return region
        except Exception as e:
            print(f"Warning: Could not create region from vertices: {str(e)}")
            # Fallback: create a simple polygon from the user points
            return reg2.RegionXY.from_vertices(user_points)

    def _create_segment_roi(self, canny_edges: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int]) -> np.ndarray:
        """
        Create a region of interest around the line segment between two points.
        Modified to better capture nearby edges.
        """
        # Create a mask for the region around the segment
        mask = np.zeros_like(canny_edges)

        # Calculate segment properties
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)

        # Create a thicker line for the ROI
        thickness = max(5, int(length / 50))
        cv.line(mask, pt1, pt2, 255, thickness)

        # Dilate the mask to include nearby edges
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv.dilate(mask, kernel, iterations=2)

        # Apply the mask to the Canny edges
        roi_edges = cv.bitwise_and(canny_edges, canny_edges, mask=dilated_mask)

        return roi_edges

    def export_regions_to_json(self, regions: dict[str, tuple], file_path: str) -> None:
        """Export multiple regions with their edges and corners to a JSON file.

        Parameters
        ----------
        regions : dict[str, tuple]
            Dictionary of regions as returned by find_multiple_regions_from_interactive_points.
            Each value should be a tuple of (edges, corners, region).
        file_path : str
            Path to save the JSON file.
        """
        import json

        # Prepare data structure for export
        export_data = {
            "regions": {},
            "metadata": {"export_format": "RegionDetector Multiple Regions", "export_version": "1.0"},
        }

        for region_name, (edges, corners, region) in regions.items():
            # Convert edges to serializable format
            edges_data = {}
            for edge_name, edge_line in edges.items():
                edges_data[edge_name] = edge_line.ABC

            # Convert corners to serializable format
            corners_data = {}
            for corner_name, corner_point in corners.items():
                corners_data[corner_name] = corner_point.data

            # Convert region to dictionary
            region_data = region.to_dict()

            export_data["regions"][region_name] = {"edges": edges_data, "corners": corners_data, "region": region_data}

        final_data = lbt.serialize_nested_dict(export_data)
        # Save to JSON file
        with open(file_path, 'w', encoding="utf-8") as f:
            json.dump(final_data, f, indent=4)

    def import_regions_from_json(self, file_path: str) -> dict[str, tuple]:
        """Import multiple regions with their edges and corners from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file.

        Returns
        -------
        dict[str, tuple]
            Dictionary of regions in the same format as returned by find_multiple_regions_from_interactive_points.
        """

        # Load from JSON file
        with open(file_path, 'r') as f:
            import_data = json.load(f)

        # Validate format
        if (
            "metadata" not in import_data
            or import_data["metadata"].get("export_format") != "RegionDetector Multiple Regions"
        ):
            raise ValueError("Invalid file format. Expected RegionDetector Multiple Regions export.")

        regions = {}

        for region_name, region_data in import_data["regions"].items():
            # Reconstruct edges
            edges = {}
            for edge_name, edge_data in region_data["edges"].items():
                edges[edge_name] = l2.LineXY(A=edge_data[0], B=edge_data[1], C=edge_data[2])

            # Reconstruct corners
            corners = {}
            for corner_name, corner_data in region_data["corners"].items():
                corners[corner_name] = p2.Pxy([corner_data[0], corner_data[1]])

            # Reconstruct region
            region = reg2.RegionXY.from_dict(region_data["region"])

            regions[region_name] = (edges, corners, region)

        return regions

    def export_masks_to_json(self, masks: list[list[tuple[int, int]]], file_path: str) -> None:
        """Export mask polygons to a JSON file.

        Parameters
        ----------
        masks : list[list[tuple[int, int]]]
            List of mask polygons, where each polygon is a list of (x,y) points.
        file_path : str
            Path to save the JSON file.
        """

        # Convert masks to serializable format
        masks_data = []
        for mask in masks:
            mask_data = [[int(point[0]), int(point[1])] for point in mask]
            masks_data.append(mask_data)

        # Prepare data structure for export
        export_data = {
            "masks": masks_data,
            "metadata": {"export_format": "RegionDetector Masks", "export_version": "1.0"},
        }

        # Save to JSON file
        with open(file_path, 'w') as f:
            json.dump(export_data, f, indent=4)

    def import_masks_from_json(self, file_path: str) -> list[list[tuple[int, int]]]:
        """Import mask polygons from a JSON file.

        Parameters
        ----------
        file_path : str
            Path to the JSON file.

        Returns
        -------
        list[list[tuple[int, int]]]
            List of mask polygons, where each polygon is a list of (x,y) points.
        """
        import json

        # Load from JSON file
        with open(file_path, 'r') as f:
            import_data = json.load(f)

        # Validate format
        if "metadata" not in import_data or import_data["metadata"].get("export_format") != "RegionDetector Masks":
            raise ValueError("Invalid file format. Expected RegionDetector Masks export.")

        # Convert masks back to original format
        masks = []
        for mask_data in import_data["masks"]:
            mask = [(point[0], point[1]) for point in mask_data]
            masks.append(mask)

        return masks

    def recreate_regions_from_export(
        self, image: np.ndarray, regions_file: str, masks_file: str = None
    ) -> dict[str, tuple]:
        """Recreate regions from exported JSON files.

        Parameters
        ----------
        regions_file : str
            Path to the regions JSON file.
        masks_file : str, optional
            Path to the masks JSON file. If provided, masks will be applied.

        Returns
        -------
        dict[str, tuple]
            Dictionary of regions in the same format as returned by find_multiple_regions_from_interactive_points.
        list[tuples]
            List of mask coordinate points as returned by find_multiple_regions_from_interactive_points.
        """
        # Load regions
        regions = self.import_regions_from_json(regions_file)

        # Load and apply masks if provided
        if masks_file:
            masks = self.import_masks_from_json(masks_file)

        return regions, masks
