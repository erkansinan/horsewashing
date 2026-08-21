"""Merkezi loglama yapilandirmasi."""
from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Uygulama genelinde kullanilacak standart logging formatini ayarlar."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Uc kutuphanelerin bilgi seviyesindeki HTTP loglarini sustur.
    # Bu satirlar uygulama hatasi degil, yalnizca istek kaydidir.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
