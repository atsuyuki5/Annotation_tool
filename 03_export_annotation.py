import json
import os
import glob
import cv2
import csv
import numpy as np
from pathlib import Path

def imread_unicode(filename):
    """日本語パスに対応した画像読み込み"""
    with open(filename, 'rb') as f:
        img_array = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def load_json(filepath):
    """JSONファイルを読み込む"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_timestamp_from_filename(filename):
    """
    ファイル名からタイムスタンプを抽出
    frame_YYYYMMDD_HHMMSS_mmm_N.png -> Unix timestamp
    """
    parts = filename.replace('.png', '').replace('.json', '').split('_')
    if len(parts) >= 4:
        # YYYYMMDD, HHMMSS, mmm を抽出
        date_str = parts[1]  # 20251124
        time_str = parts[2]  # 140455
        ms_str = parts[3]    # 948
        
        # Unix timestampを計算（簡易的に秒.ミリ秒として返す）
        # 実際のタイムスタンプ形式に合わせて調整が必要
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6])
        millisecond = int(ms_str)
        
        from datetime import datetime
        dt = datetime(year, month, day, hour, minute, second, millisecond * 1000)
        return dt.timestamp()
    return 0.0

def draw_annotations_on_image(image, shapes):
    """
    画像にアノテーションを描画
    """
    image_copy = image.copy()
    
    # 色のパレット（track_idごとに色を変える）
    colors = [
        (0, 255, 0),    # 緑
        (255, 0, 0),    # 青
        (0, 0, 255),    # 赤
        (255, 255, 0),  # シアン
        (255, 0, 255),  # マゼンタ
        (0, 255, 255),  # 黄色
    ]
    
    for shape in shapes:
        label = shape.get('label', 'unknown')
        track_id = shape.get('track_id', '0')
        points = shape.get('points', [])
        
        if len(points) >= 2:
            # バウンディングボックスを描画
            x1, y1 = int(points[0][0]), int(points[0][1])
            x2, y2 = int(points[1][0]), int(points[1][1])
            
            # track_idに応じた色を選択
            try:
                color_idx = int(track_id) % len(colors)
            except:
                color_idx = 0
            color = colors[color_idx]
            
            # バウンディングボックスを描画
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 2)
            
            # ラベルとtrack_idを描画
            text = f"{label} ID:{track_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # テキストの背景を描画
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image_copy, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(image_copy, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return image_copy

def export_video_and_csv(camera_image_dir, output_video_path, output_csv_path):
    """
    Camera_imageディレクトリから動画とCSVを生成
    """
    # 画像ファイルとJSONファイルを取得
    image_files = sorted(glob.glob(os.path.join(camera_image_dir, "frame_*.png")))
    
    if len(image_files) == 0:
        print("Error: No image files found")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # 最初の画像を読み込んで動画のサイズを取得
    first_image = imread_unicode(image_files[0])
    if first_image is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        print("Checking if file exists:", os.path.exists(image_files[0]))
        return
    height, width = first_image.shape[:2]
    
    # 動画ライターを作成
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # フレームレート
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # CSV出力用のデータを準備
    csv_data = []
    csv_headers = ['timestamp_or_frame', 'label', 'track_id', 'conf', 'x_start', 'y_start', 'x_end', 'y_end']
    
    # 各画像を処理
    for idx, image_path in enumerate(image_files):
        image_name = os.path.basename(image_path)
        json_path = image_path.replace('.png', '.json')
        
        # タイムスタンプを抽出
        timestamp = extract_timestamp_from_filename(image_name)
        
        # 画像を読み込む
        image = imread_unicode(image_path)
        
        if image is None:
            print(f"Warning: Could not read {image_path}")
            continue
        
        # JSONファイルが存在する場合、アノテーションを読み込んで描画
        if os.path.exists(json_path):
            json_data = load_json(json_path)
            shapes = json_data.get('shapes', [])
            
            # アノテーションを描画
            annotated_image = draw_annotations_on_image(image, shapes)
            
            # CSVデータに追加
            for shape in shapes:
                label = shape.get('label', 'unknown')
                track_id = shape.get('track_id', '0')
                points = shape.get('points', [])
                
                if len(points) >= 2:
                    x_start = int(points[0][0])
                    y_start = int(points[0][1])
                    x_end = int(points[1][0])
                    y_end = int(points[1][1])
                    
                    csv_data.append([
                        round(timestamp, 3),
                        label,
                        track_id,
                        1.0,  # 信頼度（アノテーションなので1.0）
                        x_start,
                        y_start,
                        x_end,
                        y_end
                    ])
        else:
            # JSONがない場合は元の画像をそのまま使用
            annotated_image = image
        
        # 動画に書き込み
        video_writer.write(annotated_image)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} frames")
    
    # 動画ライターを解放
    video_writer.release()
    print(f"Video saved to: {output_video_path}")
    
    # CSVファイルを書き込み
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        writer.writerows(csv_data)
    
    print(f"CSV saved to: {output_csv_path}")
    print(f"Total annotations: {len(csv_data)}")

if __name__ == "__main__":
    # パスの設定
    camera_image_dir = r"C:\Users\1018161-z100\Desktop\Annotation\Output\20260116150012\007-067_1\01_Camera_image_15fpsアノテ"
    
    # ディレクトリの存在確認
    if not os.path.exists(camera_image_dir):
        print(f"Error: Directory not found: {camera_image_dir}")
        exit(1)
    
    output_video_path = os.path.join(camera_image_dir, "annotation_tracking.mp4")
    output_csv_path = os.path.join(camera_image_dir, "annotation_tracking.csv")
    
    print(f"Camera image directory: {camera_image_dir}")
    print("Starting export process...")
    export_video_and_csv(camera_image_dir, output_video_path, output_csv_path)
    print("Done!")
