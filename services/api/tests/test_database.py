from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from app.database import Base, add_missing_columns
from app import models  # noqa: F401  (registers tables on Base.metadata)


class SchemaUpgradeTests(unittest.TestCase):
    def test_adds_processing_mode_to_an_existing_scans_table_and_keeps_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            engine = create_engine(f"sqlite:///{Path(temporary) / 'legacy.db'}")
            Base.metadata.create_all(engine)
            with engine.begin() as connection:  # simulate a database created before the column existed
                connection.execute(text("ALTER TABLE scans DROP COLUMN processing_mode"))
                connection.execute(text(
                    "INSERT INTO scans (id, status, progress, device_platform, device_model, capture_duration_ms, "
                    "created_at, updated_at, retention_deadline) VALUES ('legacy', 'CREATED', 0, 'ios', 'x', 1, "
                    "'2026-01-01', '2026-01-01', '2026-01-02')"
                ))
            self.assertEqual(add_missing_columns(engine), ["scans.processing_mode"])
            self.assertIn("processing_mode", {column["name"] for column in inspect(engine).get_columns("scans")})
            with engine.connect() as connection:
                mode = connection.execute(text("SELECT processing_mode FROM scans WHERE id = 'legacy'")).scalar_one()
            self.assertEqual(mode, "detailed")
            self.assertEqual(add_missing_columns(engine), [])
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
