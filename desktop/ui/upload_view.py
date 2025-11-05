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

from desktop.api_client import ApiClient
from desktop.state import AppState


class UploadWorker(QObject):
    progress = Signal(int, int)  # bytes_read, total
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, trip_id: str, file_path: str) -> None:
        super().__init__()
        self.trip_id = trip_id
        self.file_path = file_path
        self.client = ApiClient(AppState.load())

    @Slot()
    def run(self) -> None:
        try:
            result = self.client.create_job(
                trip_id=self.trip_id,
                video_path=self.file_path,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class UploadWindow(QMainWindow):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CVVR Uploader - Upload")
        self._client = ApiClient(AppState.load())

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

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.upload_btn = QPushButton("Start Upload")
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

        self.status_label.setText("Uploading...")
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

    @Slot(dict)
    def on_finished(self, result: dict) -> None:
        self._set_busy(False)
        self.status_label.setText("Upload complete. Job started.")
        try:
            job_id = result.get("job_id")
            if job_id:
                msg = f"Job started successfully.\nJob ID: {job_id}"
                QMessageBox.information(self, "Success", msg)
        except Exception:
            pass
        # Reset to allow new upload (stay on page)
        self.file_path.clear()
        self.progress.setValue(0)

    @Slot(str)
    def on_error(self, err: str) -> None:
        self._set_busy(False)
        self.status_label.setText("")
        QMessageBox.critical(self, "Upload failed", err)


