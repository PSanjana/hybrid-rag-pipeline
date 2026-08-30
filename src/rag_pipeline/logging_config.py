"""Application logging setup."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the application.

    Must be called explicitly by an entry point; importing this module has no side effects.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
