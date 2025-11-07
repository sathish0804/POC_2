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
    trips_api_url: str = "https://api.mindcoinapps.com/ai_demo_api/cvvr/cvvrTrips/report/cvvrTripsDataReport"
    # Default filters for trips report API; can be overridden via config file
    trips_filter_created_by: str = "0d7e3e26-3be6-4f43-bc70-61a8c0b312a0"
    trips_filter_div_id: str = "2c9793a1-cf8c-47d5-b55e-73023097ba3f"
    trips_filter_year_month: str = "2025-09"
    trips_filter_to_year_month: str = "2025-11"

    @classmethod
    def load(cls) -> "AppState":
        path = _config_path()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Start with defaults, then override with any provided values.
            # This ensures new keys added in later versions keep their default values
            # when older config files do not contain them.
            inst = cls()
            if "results_api_url" in data:
                inst.results_api_url = str(data.get("results_api_url") or inst.results_api_url)
            if "results_api_url_no_events" in data:
                inst.results_api_url_no_events = str(data.get("results_api_url_no_events") or inst.results_api_url_no_events)
            if "trips_api_url" in data:
                inst.trips_api_url = str(data.get("trips_api_url") or inst.trips_api_url)
            if "trips_filter_created_by" in data:
                inst.trips_filter_created_by = str(data.get("trips_filter_created_by") or inst.trips_filter_created_by)
            if "trips_filter_div_id" in data:
                inst.trips_filter_div_id = str(data.get("trips_filter_div_id") or inst.trips_filter_div_id)
            if "trips_filter_year_month" in data:
                inst.trips_filter_year_month = str(data.get("trips_filter_year_month") or inst.trips_filter_year_month)
            if "trips_filter_to_year_month" in data:
                inst.trips_filter_to_year_month = str(data.get("trips_filter_to_year_month") or inst.trips_filter_to_year_month)
            return inst
        except Exception:
            return cls()

    def save(self) -> None:
        path = _config_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "results_api_url": self.results_api_url,
                "results_api_url_no_events": self.results_api_url_no_events,
                "trips_api_url": self.trips_api_url,
                "trips_filter_created_by": self.trips_filter_created_by,
                "trips_filter_div_id": self.trips_filter_div_id,
                "trips_filter_year_month": self.trips_filter_year_month,
                "trips_filter_to_year_month": self.trips_filter_to_year_month,
            }, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


