from ui.main_ui import NoiseCancelUI
from PyQt5.QtWidgets import QApplication

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = NoiseCancelUI()
    ui.show()
    sys.exit(app.exec_())
