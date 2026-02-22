"""SQLAlchemy models for MLOps tracking and data provenance."""

from datetime import datetime
from typing import Any, List

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Experiment(Base):
    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    runs: Mapped[list["Run"]] = relationship("Run", back_populates="experiment")

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"))
    name: Mapped[str] = mapped_column(String, index=True)
    run_type: Mapped[str] = mapped_column(String)  # "GENERATION", "TRAINING", "EVAL"
    status: Mapped[str] = mapped_column(String)    # "RUNNING", "COMPLETED", "FAILED"
    
    log_file_path: Mapped[str | None] = mapped_column(String)
    raw_stdout_path: Mapped[str | None] = mapped_column(String)
    artifacts_dir: Mapped[str | None] = mapped_column(String)
    error_traceback: Mapped[str | None] = mapped_column(Text)
    
    start_time: Mapped[datetime] = mapped_column(default=datetime.now)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)

    experiment: Mapped[Experiment | None] = relationship("Experiment", back_populates="runs")
    params: Mapped[list["RunParam"]] = relationship("RunParam", back_populates="run")
    metrics: Mapped[list["RunMetric"]] = relationship("RunMetric", back_populates="run")
    records: Mapped[list["Record"]] = relationship("Record", back_populates="source_run")
    datasets_used: Mapped[list["RunDataset"]] = relationship("RunDataset", back_populates="run")

class RunParam(Base):
    __tablename__ = "run_params"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    key: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[str] = mapped_column(String)
    
    run: Mapped[Run] = relationship("Run", back_populates="params")

class RunMetric(Base):
    __tablename__ = "run_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    step: Mapped[int] = mapped_column(Integer)
    key: Mapped[str] = mapped_column(String, index=True)
    value: Mapped[float] = mapped_column(Float)
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now)

    run: Mapped[Run] = relationship("Run", back_populates="metrics")

class Record(Base):
    __tablename__ = "records"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    data_type: Mapped[str] = mapped_column(String)  # "AUDIO", "TEXT"
    file_path: Mapped[str | None] = mapped_column(String, unique=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    file_hash: Mapped[str | None] = mapped_column(String)
    is_valid: Mapped[bool] = mapped_column(default=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    source_run: Mapped[Run | None] = relationship("Run", back_populates="records")
    datasets: Mapped[list["DatasetRecord"]] = relationship("DatasetRecord", back_populates="record")

class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    records: Mapped[list["DatasetRecord"]] = relationship("DatasetRecord", back_populates="dataset")
    runs_using: Mapped[list["RunDataset"]] = relationship("RunDataset", back_populates="dataset")

class DatasetRecord(Base):
    __tablename__ = "dataset_records"
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), primary_key=True)
    record_id: Mapped[int] = mapped_column(ForeignKey("records.id"), primary_key=True)

    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="records")
    record: Mapped[Record] = relationship("Record", back_populates="datasets")

class RunDataset(Base):
    __tablename__ = "run_datasets"
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), primary_key=True)
    usage: Mapped[str] = mapped_column(String)  # "TRAIN", "EVAL", "SAFETY"

    run: Mapped[Run] = relationship("Run", back_populates="datasets_used")
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="runs_using")
