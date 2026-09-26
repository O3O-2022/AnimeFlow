"""GUI를 화면 없이 띄워 분석까지 실행하고 스크린샷을 저장 (레이아웃 확인용).

사용: python tests/gui_snapshot.py 입력영상 스크린샷.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QSettings, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from animeflow.gui.main_window import MainWindow  # noqa: E402
from animeflow.gui.style import QSS  # noqa: E402


def main():
    src, shot = sys.argv[1], sys.argv[2]
    QSettings("AnimeFlow", "AnimeFlow").remove("settings")
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = MainWindow()
    w.resize(1400, 900)
    w.show()
    w._set_input(src)
    w.lighting_check.setChecked(False)
    w.shading_check.setChecked(False)
    w.relight_check.setChecked(True)

    def after_analysis(*_):
        QTimer.singleShot(2500, lambda: (w.grab().save(shot), print("saved", shot), app.quit()))
    w._start(render=False)
    w.task.done.connect(after_analysis)
    w.task.failed.connect(lambda m: (print("FAILED", m), app.quit()))
    QTimer.singleShot(60000, app.quit)
    app.exec()


if __name__ == "__main__":
    main()
