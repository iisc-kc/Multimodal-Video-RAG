"""Logging setup for the application."""

import sys
from pathlib import Path

from loguru import logger

from .config import settings


def setup_logger():
    """Configure loguru logger with custom settings."""
    
    # Remove default logger
    logger.remove()
    
    # Console logging (with colors)
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File logging (JSON format for easier parsing)
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
    )
    
    return logger


# Initialize logger
log = setup_logger()
