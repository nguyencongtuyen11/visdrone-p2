import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from rl_sahi.common.config import load_default_config
from rl_sahi.rl.state_config import StateConfig
from rl_sahi.common.cache import (
    detection_cache_metadata, 
    detection_cache_path, 
    hard_region_cache_path,
    detection_cache_is_current,
    load_detection_cache,
    _metadata_json,
    DETECTION_CACHE_VERSION
)
from rl_sahi.common.data import iter_images
import numpy as np


# giải thích: Hàm chính kiểm tra tính tồn tại và tính hợp lệ của tệp cache phát hiện (detection cache)
def main():
    cfg = load_default_config(None, ROOT)
    state_cfg = cfg.dataclass_instance("state", StateConfig)
    detect_cfg = cfg.section("detect")
    
    # giải thích: Tạo siêu dữ liệu (metadata) kỳ vọng dựa trên các cấu hình hiện tại để đối chiếu
    expected_metadata = detection_cache_metadata(
        weights=cfg.path_value("weights"),
        imgsz=int(detect_cfg["imgsz"]),
        conf=float(detect_cfg["conf"]),
        iou=float(detect_cfg["iou"]),
        max_det=int(detect_cfg["max_det"]),
        feature_layers=cfg.feature_layers("detect"),
        aux_grid_size=int(state_cfg.grid_size),
        spatial_feature_channels=int(state_cfg.spatial_feature_channels),
    )
    
    print("Expected metadata:")
    print(_metadata_json(expected_metadata))
    print()
    
    image_root = cfg.path_value("image_root")
    cache_root = cfg.path_value("cache_root")
    split = "train"
    
    # giải thích: Lấy ra tối đa 5 ảnh để thực hiện kiểm tra
    images = list(iter_images(image_root, split=split, limit=5))
    if not images:
        print("No images found!")
        return
        
    # giải thích: Kiểm tra từng ảnh xem cache phát hiện và cache vùng khó có tồn tại và khớp metadata hay không
    for image_path in images:
        det_path = detection_cache_path(cache_root, split, image_path)
        hard_path = hard_region_cache_path(cache_root, split, image_path)
        
        print(f"Checking {image_path}:")
        print(f"  det_path exists: {det_path.exists()}")
        print(f"  hard_path exists: {hard_path.exists()}")
        
        if det_path.exists():
            with np.load(det_path, allow_pickle=False) as data:
                print("  Files in det_path:", data.files)
                if "cache_version" in data.files:
                    version = int(np.asarray(data["cache_version"]).item())
                    print(f"  version: {version} (expected >={DETECTION_CACHE_VERSION})")
                if "metadata_json" in data.files:
                    actual_meta = str(data["metadata_json"].item())
                    print(f"  actual metadata:")
                    print(f"    {actual_meta}")
                    print(f"  matches: {actual_meta == _metadata_json(expected_metadata)}")
                else:
                    print("  metadata_json missing")
                    
        print(f"  detection_cache_is_current: {detection_cache_is_current(det_path, expected_metadata)}")
        print()

if __name__ == "__main__":
    main()
