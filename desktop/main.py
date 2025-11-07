from __future__ import annotations

import os
import sys
import multiprocessing as mp
import faulthandler
from PySide6.QtWidgets import QApplication, QMessageBox

# Ensure Requests/SSL can verify HTTPS inside the bundled app
try:
    import certifi  # type: ignore
    _cafile = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", _cafile)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _cafile)
except Exception:
    pass

# Ensure project root is on sys.path when running as `python desktop/main.py`
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from desktop.ui.login_view import LoginWindow


_WINDOWS: list = []


def main() -> int:
    # Stabilize Qt + multiprocessing on macOS/Linux
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        faulthandler.enable()
    except Exception:
        pass
    app = QApplication(sys.argv)
    app.setApplicationName("CVVR Uploader")

    login = LoginWindow()

    def on_login_success() -> None:
        try:
            from desktop.ui.trips_view import TripsWindow  # type: ignore
            trips = TripsWindow()
            _WINDOWS.append(trips)
            trips.show()
            login.close()
        except Exception as e:
            QMessageBox.critical(None, "Startup error", f"Failed to open Upload screen:\n{e}")

    login.login_success.connect(on_login_success)
    login.show()
    _WINDOWS.append(login)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())


