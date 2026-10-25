"""Entry point: python main.py"""
import sys

from PyQt6.QtWidgets import QApplication

import config
from gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    app.setOrganizationName(config.ORGANIZATION_NAME)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
