from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QProgressBar,
    QMessageBox,
)

from desktop.state import AppState
from desktop.local_pipeline import process_video_locally
from desktop.results_uploader import post_results


class UploadWorker(QObject):
    progress = Signal(int, int)  # processed, total frames (sampled)
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, trip_id: str, file_path: str) -> None:
        super().__init__()
        self.trip_id = trip_id
        self.file_path = file_path

    @Slot()
    def run(self) -> None:
        try:
            events = process_video_locally(
                video_path=self.file_path,
                trip_id=self.trip_id,
                progress_cb=lambda done, total: self.progress.emit(done, total),
                sample_fps=0.5,
                verbose=False,
            )
            self.finished.emit(events)
        except Exception as e:
            self.error.emit(str(e))


class UploadWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CVVR Uploader - Upload")
        self.setFixedSize(640, 420)
        # Load state for endpoint reuse
        self._state = AppState.load()

        container = QWidget(self)
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        form = QFormLayout()
        layout.addLayout(form)

        self.trip_id = QLineEdit("")
        self.trip_id.setPlaceholderText("TRIP-001")
        form.addRow("Trip ID", self.trip_id)

        file_row = QVBoxLayout()
        self.file_path = QLineEdit("")
        self.file_path.setPlaceholderText("/path/to/video.mp4")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.on_browse)
        form.addRow("Video File", self.file_path)
        layout.addWidget(self.browse_btn)

        # Results API is enforced from defaults in AppState; no user input required

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.upload_btn = QPushButton("Start Processing")
        self.upload_btn.clicked.connect(self.on_upload)
        layout.addWidget(self.upload_btn)

        layout.addStretch(1)

        # Worker thread state
        self._thread: Optional[QThread] = None
        self._worker: Optional[UploadWorker] = None

    def on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Video", os.path.expanduser("~"), "Video Files (*.mp4 *.mov *.mkv *.avi)")
        if path:
            self.file_path.setText(path)

    def _set_busy(self, busy: bool) -> None:
        self.upload_btn.setEnabled(not busy)
        self.browse_btn.setEnabled(not busy)
        self.trip_id.setEnabled(not busy)
        self.file_path.setEnabled(not busy)

    def on_upload(self) -> None:
        trip = (self.trip_id.text() or "").strip()
        path = (self.file_path.text() or "").strip()
        if not trip:
            QMessageBox.warning(self, "Missing Trip ID", "Please enter a trip ID.")
            return
        if not path:
            QMessageBox.warning(self, "Missing File", "Please select a video file.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "Invalid File", "Selected video file does not exist.")
            return
        # Results API URL will be taken from state defaults if field is empty

        self.status_label.setText("Processing...")
        self.progress.setValue(0)
        self._set_busy(True)

        self._thread = QThread()

        self._worker = UploadWorker(trip, path)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.on_progress)
        self._worker.finished.connect(self.on_finished)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(int, int)
    def on_progress(self, done: int, total: int) -> None:
        if total > 0:
            pct = int(min(100, max(0, round((done / total) * 100))))
            self.progress.setValue(pct)
        else:
            self.progress.setValue(0)

    @Slot(list)
    def on_finished(self, events: list) -> None:
        self._set_busy(False)
        self.status_label.setText("Processing complete. Events generated.")
        try:
            # Persist results next to the input file
            out_dir = os.path.join(os.path.dirname(self.file_path.text().strip() or self.file_path.text()), "output")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "events.json")
            import json
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(events or [], f, ensure_ascii=False, indent=2)
            # Post to results endpoint using enforced defaults from AppState
            url = (self._state.results_api_url or "").strip()
            no_events = (self._state.results_api_url_no_events or "").strip()
            token = ""
            trip = (self.trip_id.text() or "").strip()
            if not url:
                QMessageBox.critical(self, "Configuration error", "Results API URL is not configured.")
            else:
                resp = post_results(trip_id=trip, events=events or [], endpoint_url=url, token=token, no_events_url=no_events)
                if resp.get("success"):
                    if resp.get("no_events"):
                        QMessageBox.information(self, "Success", f"Saved results to:\n{out_path}\nPosted no-events payload successfully.")
                    else:
                        QMessageBox.information(self, "Success", f"Saved results to:\n{out_path}\nPosted {len(events or [])} event(s) to API successfully.")
                else:
                    err = resp.get("error") or resp.get("status_code")
                    QMessageBox.critical(self, "Post failed", f"Saved results to:\n{out_path}\nAPI post failed: {err}")
        except Exception:
            pass
        # Reset to allow new processing (stay on page)
        self.file_path.clear()
        self.progress.setValue(0)

    @Slot(str)
    def on_error(self, err: str) -> None:
        self._set_busy(False)
        self.status_label.setText("")
        QMessageBox.critical(self, "Upload failed", err)


