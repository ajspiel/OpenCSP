# from opencsp.app.lookback.interactive_video_info_extract_ref_pixel import interactive_video_select
from contrib.app.LookFast.interactive_video_info_extract_ref_pixel import interactive_video_select

from opencsp.app.camera_calibration.CameraCalibration import CalibrationGUI

if __name__ == "__main__":
    video_path = r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-xx-xx_HeliostatTestLOOKFAST\1_Plan\camera_calibration_test\optics_lab_50mm_cal_target_test.MOV"
    destination_path = r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-xx-xx_HeliostatTestLOOKFAST\1_Plan\camera_calibration_test\extracted_frames"
    frame_subset_dir = r"\\snl\Collaborative\NSTTF_Optics_NDA\SolarDynamics\Experiments\2026-xx-xx_HeliostatTestLOOKFAST\1_Plan\camera_calibration_test\calibration_frames"
    start_frame = None
    end_frame = None
    reference_pixel = None

    scrubber_info = interactive_video_select(
        video_path=video_path,
        dest_path=destination_path,
        frame_subset_dir=frame_subset_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        reference_pixel=reference_pixel,
    )

    CalibrationGUI()
