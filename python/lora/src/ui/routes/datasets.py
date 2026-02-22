"""Dataset exploration routes."""

from sqlalchemy.orm import joinedload
from starlette.responses import FileResponse

from db.models import Dataset, Record
from ui.components.dataset import DatasetTable, RecordTable
from ui.components.layout import PageLayout
from ui.core import db_client


def setup_routes(app, rt):
    @rt("/datasets")
    def get():
        with db_client.session_scope() as session:
            datasets = session.query(Dataset).options(joinedload(Dataset.records)).all()
            # Just grab the last 10 audio records to preview
            recent_records = session.query(Record).order_by(Record.id.desc()).limit(10).all()

            return PageLayout(
                DatasetTable(datasets), RecordTable(recent_records), title="Datasets - LoRA Studio"
            )

    @rt("/audio/{record_id}")
    def get_audio(record_id: int):
        from starlette.responses import PlainTextResponse

        with db_client.session_scope() as session:
            record = session.query(Record).filter_by(id=record_id).first()
            if not record or not record.file_path:
                return PlainTextResponse("Not found", status_code=404)

            import os

            if os.path.exists(record.file_path):
                return FileResponse(record.file_path, media_type="audio/wav")
            from starlette.responses import PlainTextResponse

            return PlainTextResponse("File missing on disk", status_code=404)
