"""
src/database/db.py
==================
Module quản lý cơ sở dữ liệu SQLite cho hệ thống tìm người (Person Retrieval).

THIẾT KẾ BẢNG (DATABASE SCHEMA):
    1. Table 'queries': Lưu thông tin các phiên tìm kiếm
       (id, created_at, gender, upper_color, lower_color, hat, glasses, backpack, threshold)
    2. Table 'search_results': Lưu các đối tượng Target tìm thấy trong từng phiên
       (id, query_id, track_id, timestamp_str, frame_idx, score, gender, upper_color, lower_color, hat, glasses, backpack, crop_path)
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("database")


class DatabaseManager:
    """
    Quản lý kết nối và thao tác dữ liệu với SQLite.
    """

    def __init__(self, db_path: str = "results/person_retrieval.db"):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self):
        """Tạo kết nối mới với SQLite (hỗ trợ đa luồng)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # Cho phép truy cập cột bằng tên như dict
        return conn

    def _init_tables(self):
        """Khởi tạo cấu trúc các bảng nếu chưa tồn tại."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 1. Bảng lưu trữ phiên tìm kiếm
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                gender TEXT,
                upper_color TEXT,
                lower_color TEXT,
                hat INTEGER,
                glasses INTEGER,
                backpack INTEGER,
                threshold REAL
            )
        """)

        # 2. Bảng lưu trữ đối tượng target tìm thấy
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER,
                track_id INTEGER,
                timestamp_str TEXT,
                frame_idx INTEGER,
                score REAL,
                gender TEXT,
                upper_color TEXT,
                lower_color TEXT,
                hat INTEGER,
                glasses INTEGER,
                backpack INTEGER,
                crop_path TEXT,
                FOREIGN KEY (query_id) REFERENCES queries(id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"Đã kiểm tra và khởi tạo Database tại: {self.db_path}")

    def save_query(self, query_dict: dict, threshold: float) -> int:
        """
        Lưu một phiên tìm kiếm mới vào bảng queries. Trả về query_id vừa tạo.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def bool_to_int(val):
            if val is True: return 1
            if val is False: return 0
            return None

        cursor.execute("""
            INSERT INTO queries (
                created_at, gender, upper_color, lower_color,
                hat, glasses, backpack, threshold
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            created_at,
            query_dict.get("gender"),
            query_dict.get("upper_color"),
            query_dict.get("lower_color"),
            bool_to_int(query_dict.get("hat")),
            bool_to_int(query_dict.get("glasses")),
            bool_to_int(query_dict.get("backpack")),
            float(threshold)
        ))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        logger.info(f"Đã lưu Query #{query_id} vào database.")
        return query_id

    def save_target_result(self, query_id: int, target_info: dict):
        """
        Lưu thông tin 1 đối tượng target tìm thấy vào bảng search_results.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        attr = target_info.get("attributes", {})

        def bool_to_int(val):
            if val is True: return 1
            if val is False: return 0
            return None

        cursor.execute("""
            INSERT INTO search_results (
                query_id, track_id, timestamp_str, frame_idx, score,
                gender, upper_color, lower_color, hat, glasses, backpack, crop_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            query_id,
            target_info.get("track_id"),
            target_info.get("timestamp"),
            target_info.get("frame_idx"),
            float(target_info.get("score", 0.0)),
            attr.get("gender"),
            attr.get("upper_color"),
            attr.get("lower_color"),
            bool_to_int(attr.get("hat")),
            bool_to_int(attr.get("glasses")),
            bool_to_int(attr.get("backpack")),
            target_info.get("crop_path")
        ))

        conn.commit()
        conn.close()

    def get_recent_queries(self, limit: int = 15) -> list:
        """
        Lấy danh sách các phiên tìm kiếm gần đây nhất.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT q.*, COUNT(r.id) as target_count
            FROM queries q
            LEFT JOIN search_results r ON q.id = r.query_id
            GROUP BY q.id
            ORDER BY q.id DESC
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        queries = [dict(row) for row in rows]
        conn.close()
        return queries

    def get_results_by_query_id(self, query_id: int) -> list:
        """
        Lấy tất cả các đối tượng target tìm thấy của một phiên tìm kiếm cụ thể.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM search_results
            WHERE query_id = ?
            ORDER BY score DESC
        """, (query_id,))

        rows = cursor.fetchall()
        results = [dict(row) for row in rows]
        conn.close()
        return results

    def clear_all_history(self):
        """Xóa sạch toàn bộ lịch sử trong database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM search_results")
        cursor.execute("DELETE FROM queries")
        conn.commit()
        conn.close()
        logger.info("Đã dọn dẹp sạch toàn bộ lịch sử Database.")
