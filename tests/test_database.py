"""
tests/test_database.py
======================
Unit tests cho DatabaseManager (F3 - Thread Safety).
"""
import pytest
import threading
import os
import tempfile
from src.database.db import DatabaseManager


class TestDatabaseManager:
    @pytest.fixture
    def db_manager(self):
        tfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tfile.close()
        manager = DatabaseManager(db_path=tfile.name)
        yield manager
        manager.close()
        try:
            os.remove(tfile.name)
        except Exception:
            pass

    def test_save_and_retrieve_query(self, db_manager):
        query = {"gender": "Male", "upper_color": "Black", "lower_color": "Blue", "hat": False, "glasses": True, "backpack": True}
        qid = db_manager.save_query(query, threshold=0.75)
        assert qid is not None and qid > 0

        queries = db_manager.get_recent_queries(limit=5)
        assert len(queries) >= 1
        assert queries[0]["gender"] == "Male"

    def test_concurrent_writes_from_threads(self, db_manager):
        """Kiểm tra nhiều luồng ghi đồng thời an toàn với thread-local connection."""
        errors = []

        def worker(thread_idx):
            try:
                for i in range(5):
                    qid = db_manager.save_query({"gender": f"User_{thread_idx}_{i}"}, threshold=0.7)
                    db_manager.save_target_result(qid, {
                        "track_id": i,
                        "timestamp": "00:01:00",
                        "frame_idx": i * 10,
                        "score": 0.85,
                        "attributes": {"gender": "Male", "upper_color": "Red"}
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        queries = db_manager.get_recent_queries(limit=50)
        assert len(queries) == 20  # 4 threads * 5 queries
