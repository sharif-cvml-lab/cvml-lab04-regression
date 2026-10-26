import importlib.util
import os
import sys


def _check_dependencies() -> bool:
    missing = [m for m in ("PyQt6", "numpy", "cv2") if importlib.util.find_spec(m) is None]
    if missing:
        print(f"Missing required packages: {', '.join(missing)}\n"
              "Install them with:  pip install -r requirements.txt")
        return False
    return True


def main() -> int:
    if not _check_dependencies():
        return 1
    import cv2  # noqa: F401  (must be imported before the Qt plugin path is cleared)

    # OpenCV's pip wheels on Linux point Qt at their own plugin folder, which breaks PyQt6.
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    from config import APP_NAME, ORGANIZATION_NAME
    from gui import HeadPointerWindow

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)

    window = HeadPointerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
