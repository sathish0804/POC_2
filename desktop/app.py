from __future__ import annotations

import os
import sys
from PySide6.QtWidgets import QApplication, QMessageBox

# Ensure project root is on sys.path when running as `python desktop/app.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from desktop.ui.login_view import LoginWindow
from desktop.ui.upload_view import UploadWindow


# Keep strong references to top-level windows to prevent GC closing them
_WINDOWS: list = []


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CVVR Uploader")

    # Always show login first (native window, no browser)
    login = LoginWindow()

    def on_login_success() -> None:
        try:
            upload = UploadWindow()
            _WINDOWS.append(upload)
            upload.show()
            login.close()
        except Exception as e:
            QMessageBox.critical(None, "Startup error", f"Failed to open Upload screen:\n{e}")

    login.login_success.connect(on_login_success)
    login.show()
    _WINDOWS.append(login)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())


