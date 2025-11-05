from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


CONFIG_DIR_NAME = ".cvvr_desktop"
CONFIG_FILE_NAME = "config.json"


def _config_dir() -> str:
    home = os.path.expanduser("~")
    path = os.path.join(home, CONFIG_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _config_path() -> str:
    return os.path.join(_config_dir(), CONFIG_FILE_NAME)


@dataclass
class AppState:
    server_url: str = ""
    api_token: str = ""

    @classmethod
    def load(cls) -> "AppState":
        path = _config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                server_url=str(data.get("server_url", "") or ""),
                api_token=str(data.get("api_token", "") or ""),
            )
        except Exception:
            return cls()

    def save(self) -> None:
        path = _config_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "server_url": self.server_url,
                "api_token": self.api_token,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


