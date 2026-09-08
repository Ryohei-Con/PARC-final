"""学習の入口。**overlay を明示 import してから** 上流の train を呼ぶ。

上流には third-party plugin 機構があり（計画 F1）、``sys.path`` 上の
``lerobot_policy_*`` を自動 import してくれる。しかし **その import 失敗は
``except Exception: logging.exception(...)`` で握り潰される**（F2）ので、plugin 名だけに
頼ると「overlay が入っていないのに学習が始まる」事故が起きる。

そこでこの wrapper を主経路にする。overlay の import に失敗したら例外がそのまま
上がり、``assert_installed()`` が実照会で検証してから ``lerobot_train.main()`` を呼ぶ。

使い方（引数はすべて上流 ``lerobot_train.py`` にそのまま渡る）::

    python scripts/train_entry.py --dataset.type=internvla_a1_5_parc ...
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

IVLA_DIR = Path(__file__).resolve().parent.parent


def _ensure_overlay_on_syspath() -> None:
    """``lerobot_policy_parc`` を import できるようにする。"""
    path = str(IVLA_DIR)
    while path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    _ensure_overlay_on_syspath()

    # ここで失敗したら例外がそのまま上がる（F2 対策の要）。
    import lerobot_policy_parc as overlay

    summary = overlay.assert_installed()

    import lerobot

    logging.info("overlay=%s upstream=%s", overlay.__file__, lerobot.__file__)
    logging.info("overlay registrations: %s", summary["robot_types"])

    from lerobot.scripts.lerobot_train import main as lerobot_main

    lerobot_main()


if __name__ == "__main__":
    main()
