"""
src/utils/logger.py
===================
Module logging đơn giản cho toàn bộ project.
Ghi log ra console VÀ file để dễ debug.
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "results/logs", level=logging.INFO):
    """
    Tạo logger cho một module.

    INPUT:
        name (str): Tên logger, thường dùng tên module, ví dụ "detector", "tracker"
        log_dir (str): Thư mục lưu file log
        level: Mức độ log (INFO, DEBUG, WARNING, ERROR)

    OUTPUT:
        logger: Python logger object

    VÍ DỤ:
        logger = get_logger("detector")
        logger.info("YOLO model loaded successfully")
        logger.warning("Low confidence detection: 0.31")
        logger.error("Cannot open video file")
    """
    # Tạo thư mục log nếu chưa có
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Nếu logger đã có handler thì không thêm nữa (tránh duplicate)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format: [2026-08-27 20:30:15] [detector] INFO: Model loaded
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler 1: In ra console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler 2: Ghi ra file log (tên theo ngày)
    today = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{today}_{name}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
