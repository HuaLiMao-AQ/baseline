"""Baseline logging setup with quiet third-party download logs."""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

BASELINE_LOGGER_NAME = "evidenceqa_baseline"
RUN_FILE_HANDLER_NAME = "evidenceqa_baseline.run_file"

QUIET_THIRD_PARTY_LOGGERS = (
    "filelock",
    "huggingface_hub",
    "huggingface_hub.file_download",
    "huggingface_hub.utils._http",
    "urllib3",
)


def configure_run_logging(log_path: Path, *, overwrite: bool) -> logging.Logger:
    """Configure a compact per-run log file and suppress noisy retries.

    Args:
        log_path: File path for the run log.
        overwrite: Whether to truncate an existing log file.

    Returns:
        Baseline logger instance.
    """

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    warnings.filterwarnings("ignore", module=r"huggingface_hub\..*")

    for name in QUIET_THIRD_PARTY_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.ERROR)

    try:
        from huggingface_hub.utils import logging as hf_logging
    except ImportError:
        pass
    else:
        _call_optional_hf_logging_api(hf_logging, "set_verbosity_error")
        _call_optional_hf_logging_api(hf_logging, "disable_progress_bar")

    logger = logging.getLogger(BASELINE_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "name", None) == RUN_FILE_HANDLER_NAME:
            logger.removeHandler(handler)
            handler.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    handler.name = RUN_FILE_HANDLER_NAME
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    logger.addHandler(handler)
    return logger


def _call_optional_hf_logging_api(hf_logging: object, name: str) -> None:
    """Call a Hugging Face logging helper when this installed version has it."""

    function = getattr(hf_logging, name, None)
    if not callable(function):
        return
    try:
        function()
    except Exception:  # noqa: BLE001 - logging setup must not break a run.
        return
