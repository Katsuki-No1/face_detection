from __future__ import annotations

import warnings

from train_dino import main


if __name__ == "__main__":
    warnings.warn(
        "scripts/train_dino_scrfd_face.py is deprecated. Use scripts/train_dino.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise SystemExit(main())
