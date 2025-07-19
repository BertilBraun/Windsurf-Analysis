# define decorator that logs exceptions and reraises them

import logging
from settings import STANDARD_OUTPUT_DIR
from pathlib import Path
import os


def setup_logging(output_dir: Path | None = None, log_file_name=None):
    """Configure logging for the windsurfing video analysis tool."""
    if output_dir is None:
        output_dir = Path(STANDARD_OUTPUT_DIR)
    else:
        output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if log_file_name is None:
        log_file_name = 'windsurf_analysis.log'
    log_file_path = output_dir / log_file_name
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path, encoding='utf-8')],
    )
    logging.info(f'Logging to {log_file_path}')
    return logging.getLogger(__name__)


def log_and_reraise(func, *args, helpers_log_and_reraise_output_dir=None, **kwargs):
    pid = os.getpid()
    setup_logging(helpers_log_and_reraise_output_dir, f'log_and_reraise_{pid}.log')
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logging.critical(e, exc_info=True)
        raise
