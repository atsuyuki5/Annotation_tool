"""
YOLO + LiDAR統合処理パイプライン
動画/画像取得 → YOLO検出 → CVAT形式変換 → LiDARクラスタリング
"""

import os
import sys
import glob
import traceback
from datetime import datetime
from typing import Tuple, List
import pandas as pd
import cv2

# ===== 設定値 (CONFIG) =====
class Config:
    """処理全体の設定値"""
    CAMERA_MODE = 'WEBCAM'  # 'TOF' or 'WEBCAM'
    CAMERA_FPS = 20
    BASE_DIR = r"C:\Users\1018161-z100\Desktop\Annotation"
    DATA_DIR = r"C:\Users\1018161-z100\Desktop\Annotation\Data"
    
    # YOLO
    YOLO_MODEL_PATH = "YOLO/yolo12m.pt"
    YOLO_CONF_THRESHOLD = 0.55
    YOLO_IOU_THRESHOLD = 0.98
    YOLO_DYNAMIC_IOU_OFFSET = 0.018
    YOLO_DEVICE = 'auto'
    
    # LiDAR Tracking
    LIDAR_MAX_DISTANCE = 1
    LIDAR_MAX_AGE = 100
    LIDAR_MIN_HITS = 3
    LIDAR_DT = 0.025
    LIDAR_PROCESS_NOISE = 0.3
    LIDAR_MEASUREMENT_NOISE = 0.06
    LIDAR_INITIAL_COVARIANCE = 3.0
    
    # LiDAR Filtering
    LIDAR_DISTANCE_MIN = 0.1
    LIDAR_DISTANCE_MAX = 10
    LIDAR_ANGLE_MIN = -80
    LIDAR_ANGLE_MAX = 80
    
    # LiDAR Masking
    LIDAR_BACKGROUND_FRAMES = 8
    LIDAR_BACKGROUND_OFFSET = 0.02
    LIDAR_JUMP_THRESHOLD = 0.5
    LIDAR_STATIC_THRESHOLD = 0.02
    LIDAR_DYNAMIC_OFFSET = -0.01
    LIDAR_STATIC_FRAMES = 10
    
    # LiDAR Clustering
    LIDAR_EPS = 0.5
    LIDAR_MIN_SAMPLES = 5
    LIDAR_PLOT_MAX_DISTANCE = 10
    LIDAR_UPDATE_INTERVAL_MS = 25
    LIDAR_SHOW_PLOT = True


# ===== Module Imports =====
def import_camera_modules():
    """Import camera module based on mode"""
    if Config.CAMERA_MODE == 'TOF':
        from Camera.DatImageProcessor import DatImageProcessor
        return DatImageProcessor
    elif Config.CAMERA_MODE == 'WEBCAM':
        from Camera.AviImageProcessor import AviImageProsessor
        return AviImageProsessor
    else:
        raise ValueError(f"Unknown CAMERA_MODE: {Config.CAMERA_MODE}")


from YOLO.YOLOProcessor import YOLOProcessor
from LiDAR.LiDARProcessor import LiDARProcessor
from Helper.Helper import Helper, ColorPrinter


# ===== Utility Functions =====
def log_error(e: Exception, context: str = "") -> None:
    """Unified error logging"""
    ColorPrinter.print(f"Error ({context}): {type(e).__name__}: {e}", "error")
    ColorPrinter.print(traceback.format_exc(), "error")


def ensure_dir(path: str) -> str:
    """Create and return directory"""
    os.makedirs(path, exist_ok=True)
    return path


# ===== Initialization =====
def validate_and_prepare_folders() -> Tuple[List[str], List[str], str]:
    """Validate folder structure and create output directory"""
    if not os.path.exists(Config.DATA_DIR):
        ColorPrinter.print(f"Data folder not found: {Config.DATA_DIR}", "error")
        sys.exit(1)
    
    ColorPrinter.print("Extracting target folders...", "normal")
    
    data_folders = Helper.find_lidar_data_folders(Config.DATA_DIR)
    data_folders = [f for f in data_folders if not f.endswith("Movie")]
    
    if not data_folders:
        ColorPrinter.print("No target folders found", "error")
        sys.exit(1)
    
    movie_data_dir = os.path.join(Config.DATA_DIR, "Movie")
    if not os.path.exists(movie_data_dir):
        ColorPrinter.print(f"Movie folder not found: {movie_data_dir}", "error")
        sys.exit(1)
    
    if Config.CAMERA_MODE == 'TOF':
        movie_folders = Helper.find_tof_data_folders(movie_data_dir)
    else:
        movie_folders = Helper.find_avi_data_folders(movie_data_dir)
    
    is_same, diff_details = Helper.compare_folder_structures(data_folders, movie_folders)
    if not is_same:
        ColorPrinter.print(f"Folder structure mismatch:\n{diff_details}", "error")
        sys.exit(1)
    
    output_folder_name = datetime.now().strftime('%Y%m%d%H%M%S')
    output_base_dir = ensure_dir(os.path.join(Config.BASE_DIR, 'Output', output_folder_name))
    
    ColorPrinter.print(f"Processing {len(data_folders)} folders", "normal")
    return data_folders, movie_folders, output_base_dir


# ===== Camera Processing =====
def process_camera_frames(movie_folder: str, output_dir_path: str) -> str:
    """Convert camera data to images"""
    CameraClass = import_camera_modules()
    
    if Config.CAMERA_MODE == "TOF":
        ColorPrinter.print("Converting DAT to PNG...", "normal")
        png_dir = ensure_dir(os.path.join(output_dir_path, '01_TOF_image'))
        try:
            processor = CameraClass(
                mask_image_path='TOF/MaskImg.png',
                calibration_csv_path='TOF/lenseCalibration_m20_vga.csv',
                frame_rate=Config.CAMERA_FPS
            )
            processor.process_dat_files(
                dat_files_dir=movie_folder,
                output_dir=png_dir,
                output_types=['rgb'],
                create_video=False
            )
        except Exception as e:
            log_error(e, "DAT processing")
            raise
    else:  # WEBCAM
        ColorPrinter.print("Extracting AVI to PNG...", "normal")
        png_dir = ensure_dir(os.path.join(output_dir_path, '01_Camera_image'))
        avi_files = glob.glob(os.path.join(movie_folder, "*.avi"))
        if not avi_files:
            raise RuntimeError(f"AVI file not found: {movie_folder}")
        try:
            processor = CameraClass(frame_rate=Config.CAMERA_FPS)
            processor.extract_frames_from_avi(avi_file_path=avi_files[0], output_dir=png_dir)
        except Exception as e:
            log_error(e, "AVI processing")
            raise
    
    return png_dir


# ===== CVAT Export =====
def export_cvat_from_yolo(yolo_csv_path: str, timestamps_csv: str, png_dir: str, output_dir: str) -> None:
    """Export YOLO results to CVAT format (YOLO 1.1)"""
    try:
        yolo_df = pd.read_csv(yolo_csv_path)
        ts_df = pd.read_csv(timestamps_csv)
        
        ts_map = {round(row.timestamp_unix, 3): int(row.frame) 
                  for row in ts_df.itertuples(index=False)}
        
        yolo_df = yolo_df.copy()
        yolo_df["frame"] = yolo_df["timestamp_or_frame"].apply(
            lambda ts: ts_map.get(round(ts, 3))
        ).astype(int)
        
        unique_labels = sorted(set(yolo_df["label"]))
        classes = {name: idx for idx, name in enumerate(unique_labels)}
        
        png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))
        if not png_files:
            raise RuntimeError(f"PNG not found: {png_dir}")
        sample_img = cv2.imread(png_files[0])
        height, width = sample_img.shape[:2]
        
        labels_dir = ensure_dir(os.path.join(output_dir, "labels"))
        grouped = yolo_df.groupby("frame")
        for frame_id, rows in grouped:
            parts = []
            for row in rows.itertuples(index=False):
                cls_id = classes[row.label]
                x_center = ((row.x_start + row.x_end) / 2.0) / width
                y_center = ((row.y_start + row.y_end) / 2.0) / height
                w_norm = (row.x_end - row.x_start) / width
                h_norm = (row.y_end - row.y_start) / height
                
                x_center = min(1.0, max(0.0, x_center))
                y_center = min(1.0, max(0.0, y_center))
                w_norm = min(1.0, max(0.0, w_norm))
                h_norm = min(1.0, max(0.0, h_norm))
                
                parts.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
            
            label_file = os.path.join(labels_dir, f"frame{int(frame_id):06d}.txt")
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(parts))
        
        classes_file = os.path.join(output_dir, "classes.txt")
        with open(classes_file, "w", encoding="utf-8") as f:
            for name, idx in sorted(classes.items(), key=lambda x: x[1]):
                f.write(f"{name}\n")
        
        ColorPrinter.print(f"CVAT export: {labels_dir}", "normal")
        
    except Exception as e:
        log_error(e, "CVAT export")
        raise


# ===== YOLO Processing =====
def process_yolo(png_dir: str, output_dir_path: str) -> None:
    """YOLO detection and tracking"""
    ColorPrinter.print("YOLO processing...", "normal")
    
    yolo_dir = ensure_dir(os.path.join(output_dir_path, '02_YOLO'))
    yolo_csv_path = os.path.join(yolo_dir, 'yolo_tracking.csv')
    yolo_timestamps_path = os.path.join(yolo_dir, 'yolo_tracking_timestamps.csv')
    yolo_mp4_path = os.path.join(yolo_dir, 'yolo_tracking.mp4')
    
    input_png_dir = png_dir
    if Config.CAMERA_MODE == "TOF":
        input_png_dir = os.path.join(png_dir, 'RGB')
    
    try:
        yolo_processor = YOLOProcessor(
            model_path=os.path.join(Config.BASE_DIR, Config.YOLO_MODEL_PATH),
            device=Config.YOLO_DEVICE,
            frame_rate=Config.CAMERA_FPS,
            debug=False
        )
        
        yolo_processor.process_video(
            source=input_png_dir,
            csv_output_path=yolo_csv_path,
            video_output_path=yolo_mp4_path,
            confidence_threshold=Config.YOLO_CONF_THRESHOLD,
            iou_threshold=Config.YOLO_IOU_THRESHOLD,
            dynamic_iou_offset=Config.YOLO_DYNAMIC_IOU_OFFSET,
            track=True,
            count=False
        )
        
        ColorPrinter.print("CVAT export format...", "normal")
        cvat_dir = ensure_dir(os.path.join(yolo_dir, 'cvat_export'))
        export_cvat_from_yolo(yolo_csv_path, yolo_timestamps_path, input_png_dir, cvat_dir)
        
    except Exception as e:
        log_error(e, "YOLO")
        raise


# ===== LiDAR Processing =====
def process_lidar(data_folder: str, output_dir_path: str) -> None:
    """LiDAR clustering and tracking"""
    ColorPrinter.print("LiDAR processing...", "normal")
    
    lidar_dir = ensure_dir(os.path.join(output_dir_path, '03_LiDAR'))
    lidar_csv_path = os.path.join(lidar_dir, 'lidar_tracking.csv')
    lidar_mp4_path = os.path.join(lidar_dir, 'lidar_tracking.mp4')
    
    try:
        processor = LiDARProcessor(
            data_folder,
            debug=False,
            max_distance=Config.LIDAR_MAX_DISTANCE,
            max_age=Config.LIDAR_MAX_AGE,
            min_hits=Config.LIDAR_MIN_HITS,
            dt=Config.LIDAR_DT,
            process_noise=Config.LIDAR_PROCESS_NOISE,
            measurement_noise=Config.LIDAR_MEASUREMENT_NOISE,
            initial_covariance=Config.LIDAR_INITIAL_COVARIANCE
        )
        
        processor.process_all_frames_and_save(
            output_csv_path=lidar_csv_path,
            video_output_path=lidar_mp4_path,
            show_plot=Config.LIDAR_SHOW_PLOT,
            plot_max_distance=Config.LIDAR_PLOT_MAX_DISTANCE,
            update_interval_ms=Config.LIDAR_UPDATE_INTERVAL_MS,
            distance_min=Config.LIDAR_DISTANCE_MIN,
            distance_max=Config.LIDAR_DISTANCE_MAX,
            angle_min=Config.LIDAR_ANGLE_MIN,
            angle_max=Config.LIDAR_ANGLE_MAX,
            background_frames=Config.LIDAR_BACKGROUND_FRAMES,
            background_offset=Config.LIDAR_BACKGROUND_OFFSET,
            jump_threshold=Config.LIDAR_JUMP_THRESHOLD,
            static_threshold=Config.LIDAR_STATIC_THRESHOLD,
            dynamic_offset=Config.LIDAR_DYNAMIC_OFFSET,
            static_frames=Config.LIDAR_STATIC_FRAMES,
            eps=Config.LIDAR_EPS,
            min_samples=Config.LIDAR_MIN_SAMPLES
        )
        
    except Exception as e:
        log_error(e, "LiDAR")
        raise


# ===== Main Processing =====
def process_folder_pair(idx: int, total: int, data_folder: str, 
                       movie_folder: str, output_base_dir: str) -> bool:
    """Process a single folder pair"""
    try:
        ColorPrinter.print(f"\n=== ({idx+1}/{total}) {os.path.basename(data_folder)} ===", "normal")
        
        output_dir = ensure_dir(
            os.path.join(output_base_dir, os.path.basename(data_folder))
        )
        
        png_dir = process_camera_frames(movie_folder, output_dir)
        process_yolo(png_dir, output_dir)
        process_lidar(data_folder, output_dir)
        
        ColorPrinter.print(f"Done: {os.path.basename(data_folder)}", "normal")
        return True
        
    except Exception as e:
        log_error(e, f"Folder: {os.path.basename(data_folder)}")
        return False


def main():
    """Main entry point"""
    try:
        ColorPrinter.print("="*60, "normal")
        ColorPrinter.print("YOLO + LiDAR Integration Pipeline", "normal")
        ColorPrinter.print(f"Mode: {Config.CAMERA_MODE}, FPS: {Config.CAMERA_FPS}", "normal")
        ColorPrinter.print("="*60, "normal")
        
        data_folders, movie_folders, output_base_dir = validate_and_prepare_folders()
        total_folders = len(data_folders)
        
        success_count = 0
        for i, (data_folder, movie_folder) in enumerate(zip(data_folders, movie_folders)):
            if process_folder_pair(i, total_folders, data_folder, movie_folder, output_base_dir):
                success_count += 1
        
        ColorPrinter.print("="*60, "normal")
        ColorPrinter.print(f"Completed: {success_count}/{total_folders} successful", "normal")
        ColorPrinter.print(f"Output: {output_base_dir}", "normal")
        ColorPrinter.print("="*60, "normal")
        
        if success_count < total_folders:
            sys.exit(1)
        
    except KeyboardInterrupt:
        ColorPrinter.print("\nProcessing cancelled", "error")
        sys.exit(130)
    except Exception as e:
        log_error(e, "Main")
        sys.exit(1)


if __name__ == "__main__":
    main()
