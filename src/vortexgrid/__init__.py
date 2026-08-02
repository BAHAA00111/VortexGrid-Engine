import os
import logging

__version__ = "0.1.0"
__author__ = "Bahaa"

# Configure root logger for VortexGrid Engine
logger = logging.getLogger("vortexgrid")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Set default log level from environment variable or fallback to INFO
log_level = os.getenv("VORTEXGRID_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

__all__ = ["__version__", "logger"]
