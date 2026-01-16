import json
import os
import glob

def linear_interpolate(start_points, end_points, frame_ratio):
    """
    線形補間を行う関数
    frame_ratio: 0.0-1.0 (0=start, 1=end)
    """
    interpolated = []
    for start, end in zip(start_points, end_points):
        x = start[0] + (end[0] - start[0]) * frame_ratio
        y = start[1] + (end[1] - start[1]) * frame_ratio
        interpolated.append([x, y])
    return interpolated

def load_json(filepath):
    """JSONファイルを読み込む"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(filepath, data):
    """JSONファイルを保存"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def find_matching_shape(shapes, label, track_id):
    """labelとtrack_idが一致するshapeを探す"""
    for shape in shapes:
        if shape.get('label') == label and shape.get('track_id') == track_id:
            return shape
    return None

if __name__ == "__main__":
    # パスの設定
    base_dir = r"C:\Users\1018161-z100\Desktop\Annotation\Output\20260116150012\007-067_1\01_Camera_image_15fpsアノテ"
    
    # ディレクトリ内のpngファイルを取得（フレーム番号でソート）
    png_files = sorted(glob.glob(os.path.join(base_dir, "frame_*.png")))
    
    # フレーム番号→png名前のマッピングを作成
    frame_to_png = {}
    for png_path in png_files:
        png_name = os.path.basename(png_path)
        # frame_YYYYMMDD_HHMMSS_mmm_N.png から最後の数字（フレーム番号）を抽出
        frame_num = int(png_name.split('_')[-1].replace('.png', ''))
        frame_to_png[frame_num] = png_name.replace('.png', '')
    
    # ディレクトリ内のjsonファイルを取得（フレーム番号でソート）
    json_files = sorted(glob.glob(os.path.join(base_dir, "frame_*.json")))
    
    if len(json_files) < 2:
        print(f"Error: At least 2 JSON files are needed. Found {len(json_files)}")
        exit(1)
    
    print(f"Found {len(json_files)} JSON files")
    print("Interpolating frames...")
    
    # 隣同士のJSONファイル間で補間
    for idx in range(len(json_files) - 1):
        start_json = json_files[idx]
        end_json = json_files[idx + 1]
        
        # ファイル名からフレーム番号を抽出
        start_filename = os.path.basename(start_json)
        end_filename = os.path.basename(end_json)
        
        start_frame = int(start_filename.split('_')[-1].split('.')[0])
        end_frame = int(end_filename.split('_')[-1].split('.')[0])
        
        print(f"\nProcessing: {start_filename} -> {end_filename}")
        print(f"  Frame range: {start_frame} -> {end_frame}")
        
        # 補間実行
        start_data = load_json(start_json)
        end_data = load_json(end_json)
        
        image_height = start_data['imageHeight']
        image_width = start_data['imageWidth']
        total_frames = end_frame - start_frame
        
        for i in range(start_frame + 1, end_frame):
            # 中間フレームのpng名からjson名を決定
            if i in frame_to_png:
                output_base_name = frame_to_png[i]
            else:
                output_base_name = f"frame_{i:06d}"
            
            frame_ratio = (i - start_frame) / total_frames
            
            # 新しいJSONデータを構築
            new_shapes = []
            
            # startのshapeをループして、endで対応するshapeを探す
            for start_shape in start_data['shapes']:
                label = start_shape['label']
                track_id = start_shape.get('track_id')
                
                # endで同じlabelとtrack_idのshapeを探す
                end_shape = find_matching_shape(end_data['shapes'], label, track_id)
                
                if end_shape is not None:
                    # 一致するshapeが見つかった場合、pointsを補間
                    start_points = start_shape['points']
                    end_points = end_shape['points']
                    
                    # pointsの数が同じ場合のみ補間
                    if len(start_points) == len(end_points):
                        interpolated_points = linear_interpolate(start_points, end_points, frame_ratio)
                    else:
                        # pointsの数が異なる場合はstartの点を使用
                        interpolated_points = start_points
                    
                    new_shape = {
                        "label": label,
                        "points": interpolated_points,
                        "group_id": start_shape.get('group_id'),
                        "track_id": track_id,
                        "description": start_shape.get('description', ""),
                        "shape_type": start_shape.get('shape_type', "rectangle"),
                        "flags": start_shape.get('flags', {}),
                        "mask": start_shape.get('mask')
                    }
                    new_shapes.append(new_shape)
                    print(f"    Interpolated: {label} (Track: {track_id})")
                else:
                    # 一致するshapeが見つからない場合、startのshapeをそのままコピー
                    new_shapes.append(start_shape)
                    print(f"    No match for: {label} (Track: {track_id}), using start shape")
            
            new_data = {
                "version": "5.4.1",
                "flags": {},
                "shapes": new_shapes,
                "imagePath": f"{output_base_name}.png",
                "imageData": None,
                "imageHeight": image_height,
                "imageWidth": image_width
            }
            
            # ファイル名を生成して保存
            output_filename = f"{output_base_name}.json"
            output_filepath = os.path.join(base_dir, output_filename)
            save_json(output_filepath, new_data)
            print(f"  Created: {output_filename}")
    
    print("\nDone!")
