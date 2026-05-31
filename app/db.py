"""
EXPERIMENT 11 — Module 5: Database (SQLite)
============================================
WHY SQLite instead of PostgreSQL:
  PostgreSQL requires a running server, user creation, and DB setup.
  SQLite is a single file (car_detection.db) — zero config, works immediately.
  The SQLAlchemy ORM layer is identical, so swapping to PostgreSQL later
  only requires changing the DATABASE_URL string.

Table: predictions
  id         — auto-increment primary key
  filename   — original uploaded filename
  result     — "Car Detected" or "No Car"
  confidence — model confidence in percent (0.0–100.0)
  timestamp  — UTC datetime of prediction
"""

import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ── Database URL ───────────────────────────────────────────────────────────────
# SQLite: file stored in project root as car_detection.db
# To switch to PostgreSQL later:
#   DATABASE_URL = "postgresql://user:password@localhost/car_db"
DATABASE_URL = "sqlite:///car_detection.db"

# ── Engine ─────────────────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # required for SQLite + FastAPI threads
)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Model ──────────────────────────────────────────────────────────────────────
class Prediction(Base):
    __tablename__ = "predictions"

    id         = Column(Integer, primary_key=True, index=True)
    filename   = Column(String,  nullable=False)
    result     = Column(String,  nullable=False)    # "Car Detected" | "No Car"
    confidence = Column(Float,   default=0.0)       # percentage 0–100
    timestamp  = Column(DateTime, default=datetime.datetime.utcnow)

    def __repr__(self):
        return (
            f"<Prediction id={self.id} file='{self.filename}' "
            f"result='{self.result}' conf={self.confidence:.1f}%>"
        )


# ── Init Helper ────────────────────────────────────────────────────────────────
def init_db():
    """
    Create all tables if they don't exist.
    Called once at FastAPI startup in main.py.
    Safe to call multiple times — does nothing if tables already exist.
    """
    Base.metadata.create_all(bind=engine)