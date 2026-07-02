import logging

_CONFIGURED = False


def get_logger(
    name: str = "news_filterer", level: int = logging.INFO
) -> logging.Logger:
    """Return a module logger with a single shared stream handler.

    Configures logging only once so repeated calls don't attach duplicate
    handlers (which would otherwise print every message multiple times).

    Args:
        name (str): The logger name, usually the module's __name__.
        level (int): The logging level to apply.

    Returns:
        logging.Logger: The configured logger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
