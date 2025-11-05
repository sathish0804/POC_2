from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from desktop.ui.login_view import LoginWindow
from desktop.ui.upload_view import UploadWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("CVVR Uploader")

    # Always show login first (native window, no browser)
    login = LoginWindow()

    def on_login_success() -> None:
        upload = UploadWindow()
        upload.show()
        login.close()

    login.login_success.connect(on_login_success)
    login.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())


