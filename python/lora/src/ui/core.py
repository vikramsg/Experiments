"""UI core setup and configurations."""

from db.client import DBClient
from lora_training.logging_utils import get_logger, setup_logging

setup_logging()
logger = get_logger("ui")

# Global UI database client
db_client = DBClient()
db_client.init_db()
