import logging
from config import LOG_FILE, LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=getattr(logging, LOG_LEVEL),
    )

    console = logging.StreamHandler()
    console.setLevel(getattr(logging, LOG_LEVEL))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)

    logging.getLogger().addHandler(console)
