# 標準モジュール
import os
# 自作モジュール
from Matcher.Matcher import LiDARYOLOMatcher
from Helper.Helper import Helper, ColorPrinter

# 処理するデータフォルダパス(サブフォルダ以降も走査)
# BASE_DIR = "Annotation"
# DATA_DIR = r"C:\Users\0140018-Z100\source\repos\3_Project_tools\HumanDetection\2nd_evaluation\human_detection_analysis_tool\Annotation\Output\20251212144831"

BASE_DIR = r"C:\Users\1018161-z100\Desktop\Annotation"
DATA_DIR = r"C:\Users\1018161-z100\Desktop\Annotation\Output\20251226175157"

print("対象フォルダ抽出中・・・")
data_folders = Helper.find_matching_folders(DATA_DIR)

folder_count = len(data_folders)
if folder_count == 0:
    ColorPrinter.print("対象フォルダがありません", "error")
    exit

# データフォルダを処理する
for i, folder in enumerate(data_folders):
    line_break = '\n\n' if i == 0 else ''
    print(f"\n\n--- ({i+1}/{folder_count}) {folder} ---")
    
    # 出力フォルダを作成
    output_dir_path = os.path.join(folder, '04_matching')
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    # データのマッチング
    result_path = os.path.join(output_dir_path, 'matching_result.csv')
    matcher = LiDARYOLOMatcher(lidar_csv_path=os.path.join(folder, '03_LiDAR', 'lidar_tracking.csv'),
                            yolo_csv_path=os.path.join(folder, '02_YOLO', 'annotation_tracking.csv'),
                            mapping_json_path=os.path.join(BASE_DIR, "Mapping/Offset_vector_mapping.json"),
                            output_file_path=result_path,
                            max_time_diff_warning=0.1,
                            max_angle_diff_warning=20) # 左右から来た歩行データで約10°の差があるのを考慮
    if not matcher.run_all_process():
        ColorPrinter.print("マッチング失敗", "error")

ColorPrinter.print(f"\n処理完了", "normal")




