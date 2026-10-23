from sqlalchemy import create_engine, inspect, text

from app.db import Base, Database


def test_schema_creation_tolerates_a_concurrent_worker(tmp_path, monkeypatch):
    """Two server workers start on one fresh DB; the slower one must not crash."""
    url = f"sqlite:///{tmp_path / 'race.db'}"
    Database(url)  # the "winning" worker creates the table
    original = Base.metadata.create_all
    # The "losing" worker checked before the table existed, so it issues CREATE anyway.
    monkeypatch.setattr(
        Base.metadata, "create_all", lambda engine: original(engine, checkfirst=False)
    )
    db = Database(url)
    assert db.stats()["requests"] == 0


def test_migration_adds_model_column_to_old_database(tmp_path):
    path = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{path}")
    with engine.begin() as conn:  # schema from before the model column existed
        conn.execute(
            text(
                "CREATE TABLE detections (id INTEGER PRIMARY KEY, filename VARCHAR(255), "
                "source VARCHAR(16), n_cars INTEGER, max_score FLOAT, threshold FLOAT, "
                "width INTEGER, height INTEGER, latency_ms FLOAT, created_at DATETIME)"
            )
        )
    db = Database(f"sqlite:///{path}")
    assert "model" in {c["name"] for c in inspect(db.engine).get_columns("detections")}
    db.add(
        filename="a.png",
        source="api",
        model="cnn",
        n_cars=1,
        max_score=0.9,
        threshold=0.5,
        width=10,
        height=10,
        latency_ms=1.0,
    )
    assert db.recent(1)[0].model == "cnn"
