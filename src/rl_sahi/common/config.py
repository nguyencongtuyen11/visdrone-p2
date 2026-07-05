# Cho phép import các annotation nâng cao từ tương lai
from __future__ import annotations

from dataclasses import fields, is_dataclass  # Các tiện ích để làm việc với dataclass
from pathlib import Path                     # Thư viện xử lý đường dẫn
from typing import Any, TypeVar              # Type hinting

import yaml                                  # Thư viện phân tích file định dạng YAML

# Khởi tạo kiểu dữ liệu generic T phục vụ cho type hints
T = TypeVar("T")

# Định nghĩa đường dẫn gốc mặc định của dự án (đi lên 3 cấp từ file config.py trong src/rl_sahi/common/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Đường dẫn tương đối và tuyệt đối tới file cấu hình mặc định default.yaml
DEFAULT_CONFIG_RELATIVE = Path("configs") / "default.yaml"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / DEFAULT_CONFIG_RELATIVE
# Các khóa hợp lệ dùng để nhúng/include một file config khác
INCLUDE_KEYS = ("include", "includes")


class ProjectConfig:
    """
    Lớp quản lý và truy xuất cấu hình của dự án (đọc từ file YAML).
    Hỗ trợ chia nhỏ cấu hình thành các phân vùng (sections), tự động phân tích đường dẫn và khởi tạo dataclass.
    """
    def __init__(self, path: Path, root: Path) -> None:
        self.path = Path(path)   # Đường dẫn tới file config
        self.root = Path(root)   # Đường dẫn thư mục gốc dự án dùng làm mốc cho các đường dẫn tương đối
        self.data = load_yaml_config(self.path) # Phân tích cú pháp file YAML

    def section(self, name: str) -> dict[str, Any]:
        """
        Lấy một phân vùng (section) cụ thể dưới dạng dictionary.
        """
        value = self.data.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section [{name}] must be a mapping")
        return dict(value)

    def path_value(self, key: str) -> Path:
        """
        Lấy giá trị của một đường dẫn từ phân vùng [paths].
        Tự động chuyển đổi thành đường dẫn tuyệt đối dựa trên self.root nếu giá trị là đường dẫn tương đối.
        """
        value = self.section("paths")[key]
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else self.root / path

    def optional_str(self, section: str, key: str) -> str | None:
        """
        Lấy một giá trị chuỗi (string) tùy chọn từ một phân vùng. Trả về None nếu không tồn tại hoặc rỗng.
        """
        value = self.section(section).get(key)
        if value is None or value == "":
            return None
        return str(value)

    def feature_layers(self, section: str) -> tuple[int, ...]:
        """
        Lấy danh sách các lớp trích xuất đặc trưng mạng nơ-ron (feature layers) từ cấu hình.
        Ví dụ: "10,12" hoặc [10, 12] -> (10, 12)
        """
        value = self.section(section).get("feature_layers", [10])
        if isinstance(value, str):
            return tuple(int(x.strip()) for x in value.split(",") if x.strip())
        return tuple(int(x) for x in value)

    def dataclass_kwargs(self, section: str, cls: type[T]) -> dict[str, Any]:
        """
        Lọc các tham số trong phân vùng cấu hình chỉ giữ lại những trường hợp lệ phù hợp với dataclass `cls`.
        """
        if not is_dataclass(cls):
            raise TypeError(f"{cls} must be a dataclass type")
        allowed = {field.name for field in fields(cls)}
        values = self.section(section)
        return {key: value for key, value in values.items() if key in allowed}

    def dataclass_instance(self, section: str, cls: type[T]) -> T:
        """
        Khởi tạo trực tiếp một đối tượng của dataclass `cls` từ phân vùng cấu hình tương ứng.
        """
        return cls(**self.dataclass_kwargs(section, cls))


def load_yaml_config(path: Path) -> dict[str, Any]:
    """
    Phân tích cú pháp tệp YAML từ đường dẫn được chỉ định.
    """
    return _load_yaml_config(path.resolve(), stack=())


def _load_yaml_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    """
    Hàm đệ quy thực tế để tải cấu hình YAML, hỗ trợ nhúng tệp tin cấu hình khác (include).
    Có cơ chế phát hiện lặp đệ quy vô hạn (circular include).
    """
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Circular config include detected: {chain}")
    data = _read_yaml_mapping(path)
    # Lấy ra danh sách các file cấu hình được nhúng (include) và xóa trường include khỏi dữ liệu hiện tại
    include_values = _pop_include_values(data, path)

    merged: dict[str, Any] = {}
    # Lần lượt tải đệ quy các cấu hình được nhúng và gộp đè lên nhau
    for include_value in include_values:
        include_path = Path(str(include_value)).expanduser()
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        merged = _deep_merge(merged, _load_yaml_config(include_path.resolve(), (*stack, path)))

    # Gộp cấu hình hiện tại đè lên trên cấu hình đã gộp từ các file nhúng
    return _deep_merge(merged, data)


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """
    Đọc tệp tin YAML và kiểm tra xem nội dung gốc có phải là dạng mapping (dictionary) hay không.
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return dict(data)


def _pop_include_values(data: dict[str, Any], path: Path) -> list[str]:
    """
    Tìm kiếm, trích xuất và xóa trường "include" hoặc "includes" trong dictionary cấu hình thô.
    """
    present = [key for key in INCLUDE_KEYS if key in data]
    if len(present) > 1:
        raise ValueError(f"Config file {path} must use only one of {INCLUDE_KEYS}")
    if not present:
        return []

    raw = data.pop(present[0])
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        return [str(raw)]
    if isinstance(raw, list):
        if not all(isinstance(value, (str, Path)) for value in raw):
            raise ValueError(f"Config include list in {path} must contain only strings")
        return [str(value) for value in raw]
    raise ValueError(f"Config include in {path} must be a string or list of strings")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Gộp đệ quy sâu (deep merge) hai dictionary. Cấu hình trong `override` sẽ đè lên `base`.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, root: Path) -> ProjectConfig:
    """
    Tạo và tải đối tượng ProjectConfig từ đường dẫn file và thư mục gốc.
    """
    return ProjectConfig(path=path, root=root)


def resolve_config_path(path: Path | str | None = None, root: Path | str | None = None) -> tuple[Path, Path]:
    """
    Xác định chính xác đường dẫn tệp tin cấu hình tuyệt đối và thư mục gốc dự án từ các tham số dòng lệnh tùy chọn.
    """
    project_root = Path(root).resolve() if root is not None else PROJECT_ROOT
    config_path = project_root / DEFAULT_CONFIG_RELATIVE if path is None else Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    return config_path, project_root


def load_default_config(path: Path | str | None = None, root: Path | str | None = None) -> ProjectConfig:
    """
    Hàm tiện ích chính để tải cấu hình mặc định của dự án.
    """
    config_path, project_root = resolve_config_path(path, root)
    return load_config(config_path, project_root)

