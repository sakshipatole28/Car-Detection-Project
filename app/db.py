"""Detection history (SQLite by default; any SQLAlchemy URL via CARDET_DB_URL)."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class DetectionLog(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(16), default="api")  # "ui" | "api"
    model: Mapped[str] = mapped_column(String(8), default="hog")  # "cnn" | "hog"
    n_cars: Mapped[int] = mapped_column(Integer)
    max_score: Mapped[float] = mapped_column(Float, default=0.0)
    threshold: Mapped[float] = mapped_column(Float)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.filename,
            "source": self.source,
            "model": self.model,
            "n_cars": self.n_cars,
            "max_score": round(self.max_score, 4),
            "threshold": self.threshold,
            "width": self.width,
            "height": self.height,
            "latency_ms": round(self.latency_ms, 1),
            "created_at": self.created_at.isoformat(timespec="seconds")
            if self.created_at
            else None,
        }


class Database:
    def __init__(self, url: str):
        kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, **kwargs)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a database was first created."""
        cols = {c["name"] for c in inspect(self.engine).get_columns("detections")}
        if "model" not in cols:
            with self.engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE detections ADD COLUMN model VARCHAR(8) DEFAULT 'hog'")
                )

    def add(self, **fields) -> DetectionLog:
        with self.Session.begin() as s:
            row = DetectionLog(**fields)
            s.add(row)
        return row

    def recent(self, limit: int = 50) -> list[DetectionLog]:
        with Session(self.engine) as s:
            stmt = select(DetectionLog).order_by(DetectionLog.id.desc()).limit(limit)
            return list(s.scalars(stmt))

    def stats(self) -> dict:
        with Session(self.engine) as s:
            total, cars, with_car, latency = s.execute(
                select(
                    func.count(DetectionLog.id),
                    func.coalesce(func.sum(DetectionLog.n_cars), 0),
                    func.coalesce(func.sum((DetectionLog.n_cars > 0).cast(Integer)), 0),
                    func.avg(DetectionLog.latency_ms),
                )
            ).one()
        return {
            "requests": total,
            "cars_found": int(cars),
            "images_with_cars": int(with_car),
            "avg_latency_ms": round(latency, 1) if latency is not None else None,
        }
