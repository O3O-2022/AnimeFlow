"""다크 테마 스타일시트."""

ACCENT = "#8b7cff"
OK = "#4cc38a"
MUTED = "#5b6172"
WARN = "#e5a44b"

QSS = f"""
* {{
    font-family: "Malgun Gothic", "Segoe UI", sans-serif;
    font-size: 10pt;
    color: #e6e8ef;
}}
QMainWindow, QWidget#root {{ background: #14161c; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}

QGroupBox {{
    background: #1c1f27;
    border: 1px solid #2a2e3a;
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 12px 10px 12px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT};
    font-size: 10.5pt;
}}
QGroupBox#experimental {{ border: 1px dashed #6b5a2e; }}
QGroupBox#experimental::title {{ color: {WARN}; }}
QLabel#hint {{ color: #8a90a2; font-size: 9pt; font-weight: normal; }}
QLabel#title {{ font-size: 17pt; font-weight: bold; color: #ffffff; }}
QLabel#subtitle {{ color: #8a90a2; }}
QLabel#statValue {{ font-size: 15pt; font-weight: bold; color: #ffffff; }}
QLabel#statName {{ color: #8a90a2; font-size: 9pt; }}
QFrame#statCard {{ background: #1c1f27; border: 1px solid #2a2e3a; border-radius: 10px; }}
QFrame#stageChip {{ background: #1c1f27; border: 1px solid #2a2e3a; border-radius: 10px; }}
QFrame#stageChip[state="run"] {{ border: 1px solid {ACCENT}; background: #221e3a; }}
QFrame#stageChip[state="done"] {{ border: 1px solid {OK}; }}
QFrame#stageChip[state="fail"] {{ border: 1px solid #d05060; }}
QLabel#stageTitle {{ font-weight: bold; }}
QLabel#stageState {{ color: #aab0c0; font-size: 9pt; }}
QLabel#caption {{ background: #1c1f27; border-radius: 6px; padding: 6px; font-size: 10.5pt; }}
QLabel#preview {{ background: #0e1015; border: 1px solid #2a2e3a; border-radius: 8px; color: #5b6172; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: #12141a;
    border: 1px solid #2f3442;
    border-radius: 6px;
    padding: 5px 8px;
    min-height: 20px;
    font-weight: normal;
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: #1c1f27; selection-background-color: {ACCENT}; border: 1px solid #2f3442; }}
QComboBox::drop-down {{ border: none; width: 22px; }}

QPushButton {{
    background: #262a35;
    border: 1px solid #343949;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: normal;
}}
QPushButton:hover {{ background: #2f3442; }}
QPushButton:disabled {{ color: #5b6172; background: #1c1f27; }}
QPushButton#primary {{
    background: {ACCENT}; border: none; color: white; font-weight: bold; font-size: 11pt; padding: 10px 26px;
}}
QPushButton#primary:hover {{ background: #9d90ff; }}
QPushButton#primary:disabled {{ background: #3b3766; color: #8a86b8; }}
QPushButton#secondary {{ font-size: 11pt; padding: 10px 20px; }}
QPushButton#danger {{ background: #3a2026; border-color: #5a2a33; font-size: 11pt; padding: 10px 20px; }}
QPushButton#danger:hover {{ background: #4a2830; }}
QPushButton#preset {{ padding: 4px 4px; min-width: 28px; }}
QPushButton#preset:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}

QCheckBox {{ font-weight: normal; spacing: 8px; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 4px; border: 1px solid #454b5e; background: #12141a; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: none; }}

QSlider::groove:horizontal {{ height: 6px; background: #2a2e3a; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 16px; margin: -6px 0; border-radius: 8px; }}
QSlider::groove:horizontal:disabled {{ background: #22252e; }}
QSlider::sub-page:horizontal:disabled {{ background: #3b3766; }}

QProgressBar {{
    background: #12141a; border: 1px solid #2a2e3a; border-radius: 7px; height: 16px; text-align: center;
    font-weight: bold;
}}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 6px; }}

QTabWidget::pane {{ border: 1px solid #2a2e3a; border-radius: 10px; background: #181a21; top: -1px; }}
QTabBar::tab {{
    background: transparent; padding: 8px 18px; color: #8a90a2; border: none; font-weight: bold;
}}
QTabBar::tab:selected {{ color: white; border-bottom: 2px solid {ACCENT}; }}

QTableWidget {{
    background: #14161c; alternate-background-color: #181b22; gridline-color: #232733;
    border: 1px solid #2a2e3a; border-radius: 8px; selection-background-color: #3b3766;
}}
QHeaderView::section {{ background: #1c1f27; color: #aab0c0; border: none; border-bottom: 1px solid #2a2e3a; padding: 6px; }}
QPlainTextEdit {{ background: #0e1015; border: none; font-family: Consolas, "Malgun Gothic"; font-size: 9pt; }}
QToolTip {{ background: #262a35; color: white; border: 1px solid {ACCENT}; padding: 4px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #2f3442; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
"""
