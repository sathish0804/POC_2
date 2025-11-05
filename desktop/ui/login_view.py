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


class LoginWindow(QMainWindow):
    login_success = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("CVVR Uploader - Login")
        self._state = AppState.load()
        self.setFixedSize(420, 240)

        container = QWidget(self)
        self.setCentralWidget(container)

        layout = QVBoxLayout(container)
        form = QFormLayout()
        layout.addLayout(form)

        self.username = QLineEdit("")
        self.username.setPlaceholderText("username")
        form.addRow("Username", self.username)

        self.password = QLineEdit("")
        self.password.setPlaceholderText("password")
        self.password.setEchoMode(QLineEdit.Password)
        form.addRow("Password", self.password)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.status_label)

        self.save_btn = QPushButton("Login")
        self.save_btn.clicked.connect(self.on_login_clicked)
        layout.addWidget(self.save_btn)

        layout.addStretch(1)

    def on_login_clicked(self) -> None:
        # Hardcoded credentials for now
        HARD_USER = "cvvr"
        HARD_PASS = "cvvr123"

        user = (self.username.text() or "").strip()
        pwd = (self.password.text() or "").strip()
        if not user or not pwd:
            QMessageBox.warning(self, "Missing credentials", "Please enter username and password.")
            return

        self.save_btn.setEnabled(False)
        try:
            if user == HARD_USER and pwd == HARD_PASS:
                self.status_label.setText("Login successful")
                self.login_success.emit()
            else:
                QMessageBox.critical(self, "Login failed", "Invalid username or password.")
        finally:
            self.save_btn.setEnabled(True)


