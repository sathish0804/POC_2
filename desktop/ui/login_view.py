from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QMessageBox,
)

from desktop.state import AppState
from desktop.api_client import ApiClient


class LoginWindow(QMainWindow):
    login_success = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CVVR Uploader - Login")
        self._state = AppState.load()

        container = QWidget(self)
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)
        form = QFormLayout()
        layout.addLayout(form)

        self.server_url = QLineEdit(self._state.server_url)
        self.server_url.setPlaceholderText("https://your-server:8000")
        form.addRow("Server URL", self.server_url)

        self.api_token = QLineEdit(self._state.api_token)
        self.api_token.setPlaceholderText("Optional API token")
        self.api_token.setEchoMode(QLineEdit.Password)
        form.addRow("API Token", self.api_token)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.status_label)

        self.save_btn = QPushButton("Login")
        self.save_btn.clicked.connect(self.on_login_clicked)
        layout.addWidget(self.save_btn)

        layout.addStretch(1)

    def on_login_clicked(self) -> None:
        url = (self.server_url.text() or "").strip()
        token = (self.api_token.text() or "").strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Please enter the server URL.")
            return

        self.status_label.setText("Checking server...")
        self.save_btn.setEnabled(False)
        try:
            state = AppState(server_url=url, api_token=token)
            client = ApiClient(state)
            _ = client.health()
            state.save()
            self.status_label.setText("OK")
            self.login_success.emit()
        except Exception as e:
            self.status_label.setText("")
            QMessageBox.critical(self, "Login failed", f"Could not reach server:\n{e}")
        finally:
            self.save_btn.setEnabled(True)


