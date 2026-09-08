import yaml
import os
from src.utils.logger import get_logger

logger = get_logger("config_loader")

def load_config(path: str = "config/config.yaml") -> dict:
    """
    Đọc cấu hình từ file YAML.
    Nếu không tìm thấy file, trả về dict rỗng để hệ thống dùng giá trị mặc định.
    """
    if not os.path.exists(path):
        logger.warning(f"Không tìm thấy file config tại {path}, sẽ dùng thông số mặc định.")
        return {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            logger.info(f"Đã tải cấu hình từ {path}")
            return config if config else {}
    except Exception as e:
        logger.error(f"Lỗi khi đọc file cấu hình {path}: {e}")
        return {}
