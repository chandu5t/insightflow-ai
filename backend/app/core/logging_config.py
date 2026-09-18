"""Central logging configuration."""

import logging


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger. Unknown level names fall back to INFO."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )