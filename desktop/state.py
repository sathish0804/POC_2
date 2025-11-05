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
    results_api_url: str = "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulk"
    results_api_url_no_events: str = "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTripViolations/addUpdateBulkNoEvents"

    @classmethod
    def load(cls) -> "AppState":
        path = _config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                results_api_url=str(data.get("results_api_url", "") or ""),
                results_api_url_no_events=str(data.get("results_api_url_no_events", "") or ""),
            )
        except Exception:
            return cls()

    def save(self) -> None:
        path = _config_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "results_api_url": self.results_api_url,
                "results_api_url_no_events": self.results_api_url_no_events,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


