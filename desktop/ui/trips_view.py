from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional

import requests
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)

from desktop.state import AppState


TripRow = Dict[str, Any]


class TripsWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CVVR Uploader - Trips")
        self.setMinimumSize(900, 520)

        self._state = AppState.load()
        self._children: List[QMainWindow] = []

        container = QWidget(self)
        self.setCentralWidget(container)
        root = QVBoxLayout(container)

        # Header bar
        header = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_trips)
        header.addWidget(self.status_label, 1)
        header.addWidget(self.refresh_btn, 0)
        root.addLayout(header)

        # Table
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Date",
            "Loco No",
            "Section",
            "ALP Name",
            "Created By",
            "Created Date",
            "Action",
        ])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 90)
        self.table.setColumnWidth(2, 130)
        self.table.setColumnWidth(3, 220)
        self.table.setColumnWidth(4, 140)
        self.table.setColumnWidth(5, 160)
        self.table.setColumnWidth(6, 100)
        root.addWidget(self.table, 1)

        # Initial load
        self.load_trips()

    def _extract_rows(self, root: Any) -> List[TripRow]:
        try:
            if isinstance(root, dict):
                # Common shape: { status, content: { totalRecords, data: [...] } }
                content = root.get("content")
                if isinstance(content, dict):
                    data = content.get("data")
                    if isinstance(data, list):
                        return [r for r in data if isinstance(r, dict)]
                # Sometimes data might be directly on root
                data = root.get("data")
                if isinstance(data, list):
                    return [r for r in data if isinstance(r, dict)]
                # If content itself is a list
                if isinstance(content, list):
                    return [r for r in content if isinstance(r, dict)]
            if isinstance(root, list):
                return [r for r in root if isinstance(r, dict)]
        except Exception:
            pass
        return []

    def load_trips(self) -> None:
        self.refresh_btn.setEnabled(False)
        self.status_label.setText("Loading trips...")
        self.table.setRowCount(0)
        url = (self._state.trips_api_url or "").strip()
        if not url:
            QMessageBox.critical(self, "Configuration error", "Trips API URL is not configured.")
            self.status_label.setText("Trips API URL missing")
            self.refresh_btn.setEnabled(True)
            return

        try:
            # Build payload from provided spec with dynamic yearMonth range
            start_ym = (self._state.trips_filter_year_month or "").strip()
            end_ym = (self._state.trips_filter_to_year_month or "").strip()

            payload = {
                "type": "yearMonthRange",
                "fromDate": "",
                "yearMonth": start_ym,
                "toYearMonth": end_ym,
                "toDate": "",
                "createdBy": (self._state.trips_filter_created_by or "").strip(),
                "crewMemberId": "",
                "divId": (self._state.trips_filter_div_id or "").strip(),
                "fromStationIds": [],
                "page": 1,
                "sectionIds": [],
                "size": 100,
                "toStationIds": [],
                # Often present but optional
                "groupByCli": 0,
                "groupByApp": 0,
                "analysisType": 0,
            }

            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            # If server rejects POST entirely, fallback to GET once
            if resp.status_code == 405:
                resp = requests.get(url, headers={"Accept": "application/json"}, timeout=20)
            resp.raise_for_status()

            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text}
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Failed to load trips: {e}")
            self.status_label.setText("Failed to load trips")
            self.refresh_btn.setEnabled(True)
            return

        try:
            rows: List[TripRow] = self._extract_rows(data)
        except Exception:
            rows = []

        self._populate_rows(rows)
        total = len(rows)
        # Provide a hint if server responded OK but no rows found
        if total == 0:
            peek = ""
            try:
                import json as _json
                peek = _json.dumps((data or {}), ensure_ascii=False)[:400]
            except Exception:
                peek = str(data)[:400]
            self.status_label.setText("No trips found for current filters")
            QMessageBox.information(self, "No Results", f"Server returned 200 but no rows.\nAbridged response:\n{peek}")
        else:
            self.status_label.setText(f"Loaded {total} trip(s)")
        self.refresh_btn.setEnabled(True)

    def _populate_rows(self, rows: List[TripRow]) -> None:
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            date_time = str(row.get("dateTime", ""))
            loco_no = str(row.get("locoNo", ""))
            section = str(row.get("sectionName", ""))
            alp_cli = str(row.get("alpCliName", ""))
            created_cli = str(row.get("createdCliName", ""))
            created_date = str(row.get("createdDate", ""))

            self.table.setItem(r, 0, QTableWidgetItem(date_time))
            self.table.setItem(r, 1, QTableWidgetItem(loco_no))
            self.table.setItem(r, 2, QTableWidgetItem(section))
            self.table.setItem(r, 3, QTableWidgetItem(alp_cli))
            self.table.setItem(r, 4, QTableWidgetItem(created_cli))
            self.table.setItem(r, 5, QTableWidgetItem(created_date))

            upload_btn = QPushButton("Upload")
            upload_btn.clicked.connect(functools.partial(self._on_upload_clicked, row))
            self.table.setCellWidget(r, 6, upload_btn)

    def _on_upload_clicked(self, row: TripRow) -> None:
        try:
            from desktop.ui.upload_view import UploadWindow  # Lazy import
        except Exception as e:
            QMessageBox.critical(self, "Navigation error", f"Cannot open upload view: {e}")
            return

        trip_id = str(row.get("uuid") or row.get("trainNo") or "").strip()
        win = UploadWindow()
        if trip_id:
            try:
                win.trip_id.setText(trip_id)
            except Exception:
                pass
        self._children.append(win)
        win.show()


