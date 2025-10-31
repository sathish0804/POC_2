import os
import sys


def resource_path(relative_path: str) -> str:
    """Resolve a data file path when running from source or PyInstaller.

    - Returns absolute path to the first existing candidate.
    - Falls back to the original relative path if nothing is found.
    """
    if not relative_path:
        return relative_path
    if os.path.isabs(relative_path):
        return relative_path

    candidates = []
    # 1) PyInstaller onefile temp folder
    try:
        candidates.append(os.path.join(sys._MEIPASS, relative_path))  # type: ignore[attr-defined]
    except Exception:
        pass

    # 2) Directory of the frozen executable
    try:
        if getattr(sys, "frozen", False):
            candidates.append(os.path.join(os.path.dirname(sys.executable), relative_path))
    except Exception:
        pass

    # 3) Project root (two levels up from this file)
    try:
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates.append(os.path.join(pkg_root, relative_path))
    except Exception:
        pass

    # 4) Current working directory
    try:
        candidates.append(os.path.join(os.getcwd(), relative_path))
    except Exception:
        pass

    for cand in candidates:
        try:
            if os.path.isfile(cand):
                return cand
        except Exception:
            continue

    return relative_path


