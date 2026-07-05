from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_sahi.common.cache import (
    DetectionCache,
    HardRegionCache,
    detection_cache_is_current,
    detection_cache_path,
    hard_region_cache_path,
    load_detection_cache,
    load_hard_region_cache,
)
from rl_sahi.common.data import iter_images


# giải thích: Lớp dữ liệu CachedSample lưu giữ đường dẫn và các đối tượng cache đã tải cho một mẫu ảnh cụ thể
@dataclass(slots=True)
class CachedSample:
    image_path: Path
    detection_path: Path
    hard_region_path: Path
    detection: DetectionCache | None = None
    hard_region: HardRegionCache | None = None


# giải thích: Lớp CachedEpisodeDataset quản lý tập hợp các mẫu huấn luyện/đánh giá sử dụng các tệp cache đã tạo trước
class CachedEpisodeDataset:
    def __init__(
        self,
        image_root: Path,
        cache_root: Path,
        split: str,
        limit: int | None = None,
        preload: bool = False,
        detection_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.image_root = Path(image_root)
        self.cache_root = Path(cache_root)
        self.split = split
        self.samples = []
        self.total_images = 0
        self.missing_detection = 0
        self.stale_detection = 0
        self.missing_hard_region = 0
        
        # giải thích: Lặp qua danh sách ảnh và kiểm tra tính hợp lệ, tính mới của tệp cache phát hiện và cache vùng khó
        for image_path in iter_images(self.image_root, split=split, limit=limit):
            self.total_images += 1
            det_path = detection_cache_path(self.cache_root, split, image_path)
            hard_path = hard_region_cache_path(self.cache_root, split, image_path)
            det_exists = det_path.exists()
            det_current = det_exists and detection_cache_is_current(det_path, detection_metadata)
            hard_exists = hard_path.exists()
            
            # giải thích: Chỉ sử dụng các mẫu có cả hai tệp cache phát hiện (mới nhất) và cache vùng khó tồn tại
            if det_current and hard_exists:
                detection = load_detection_cache(det_path) if preload else None
                hard_region = load_hard_region_cache(hard_path) if preload else None
                self.samples.append(CachedSample(image_path, det_path, hard_path, detection, hard_region))
            else:
                if not det_exists:
                    self.missing_detection += 1
                elif not det_current:
                    self.stale_detection += 1
                if not hard_exists:
                    self.missing_hard_region += 1
                    
        # giải thích: Báo lỗi nếu không tìm thấy mẫu cache hợp lệ nào
        if not self.samples:
            raise FileNotFoundError(
                f"No paired current detection/hard-region caches found for split '{split}'. "
                "Run scripts/detect.py and scripts/hard_region.py first."
            )
        if self.total_images != len(self.samples):
            print(
                f"[dataset] split={split} using {len(self.samples)}/{self.total_images} cached episodes "
                f"(missing_detection={self.missing_detection}, stale_detection={self.stale_detection}, "
                f"missing_hard_region={self.missing_hard_region})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    # giải thích: Chọn một tập ngẫu nhiên và tải cache của nó lên để phục vụ vòng huấn luyện tiếp theo
    def random_episode(self):
        sample = random.choice(self.samples)
        if sample.detection is not None and sample.hard_region is not None:
            return sample.detection, sample.hard_region
        return load_detection_cache(sample.detection_path), load_hard_region_cache(sample.hard_region_path)

    # giải thích: Lấy cache phát hiện đầu tiên để trích xuất các tham số kích thước không gian trạng thái ban đầu
    def first_detection(self):
        sample = self.samples[0]
        if sample.detection is not None:
            return sample.detection
        return load_detection_cache(sample.detection_path)
