"""Shared dark desktop palette."""
STYLE = """
            QWidget { background: #16191f; color: #dce1e9; font-family: "Noto Sans CJK SC", "DejaVu Sans", sans-serif; font-size: 13px; }
            QLabel { background: transparent; }
            QLabel#appTitle { font-size: 20px; font-weight: 600; }
            QLabel#sectionTitle { font-size: 14px; font-weight: 600; }
            QLabel#muted { color: #939ba9; }
            QLabel#metricValue { font-size: 25px; font-weight: 600; }
            QLabel#sampleCount { font-size: 42px; font-weight: 600; }
            QLabel#modeBadge { background: #252c38; color: #a8c9f8; border-radius: 4px; padding: 4px 9px; }
            QFrame#sidebar, QFrame#metricCard, QFrame#safetyBar, QFrame#qualityGate { background: #1e232c; border: 1px solid #2b313c; border-radius: 7px; }
            QLabel#imagePreview { background: #101318; color: #7c8595; border: 1px solid #2b313c; border-radius: 7px; }
            QPushButton { background: #242a34; border: 1px solid #343c49; border-radius: 5px; padding: 8px 14px; }
            QPushButton:hover { background: #303949; border-color: #658cc2; }
            QPushButton:focus { border-color: #79afff; }
            QPushButton:disabled { background: #1d222a; color: #667080; border-color: #2b313c; }
            QPushButton#primary { background: #3676d8; color: white; border-color: #3676d8; font-weight: 600; }
            QPushButton#primary:hover { background: #4387ed; }
            QPushButton#primary:disabled { background: #263d5e; color: #8a9bb5; border-color: #263d5e; }
            QPushButton#nav, QPushButton#viewSwitch { background: transparent; border-color: transparent; color: #939ba9; }
            QPushButton#nav:checked, QPushButton#viewSwitch:checked { background: #29364a; color: #a8c9f8; }
            QLineEdit, QComboBox { background: #191d24; border: 1px solid #343c49; border-radius: 4px; padding: 7px 9px; }
            QLineEdit:disabled, QComboBox:disabled, QCheckBox:disabled { color: #667080; }
            QComboBox QAbstractItemView, QMenu { background: #242a34; border: 1px solid #343c49; padding: 5px; }
            QMenu::item { padding: 8px 18px; }
            QMenu::item:selected { background: #314a6e; }
            QMenu::item:disabled { color: #667080; }
            QTableWidget { background: #191d24; alternate-background-color: #1d222a; gridline-color: #2b313c; border: none; selection-background-color: #2a4264; }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section { background: #222833; color: #a5afbf; padding: 9px 6px; border: none; border-bottom: 1px solid #343c49; }
            QTabWidget::pane { border: 1px solid #2b313c; border-radius: 5px; }
            QTabBar::tab { padding: 9px 22px; color: #939ba9; border-bottom: 2px solid transparent; }
            QTabBar::tab:selected { color: #a8c9f8; border-bottom-color: #609bec; }
            QProgressBar { background: #12161d; border: 1px solid #303846; border-radius: 5px; text-align: center; min-height: 22px; }
            QProgressBar::chunk { background: #315e99; border-radius: 4px; }
            QScrollArea { border: none; background: transparent; }
            QWidget#deviceSummary, QScrollArea#deviceScroll, QScrollArea#deviceScroll > QWidget { background: #1e232c; }
            QTextEdit { background: #191d24; border: none; padding: 12px; }
            QStatusBar { color: #939ba9; font-size: 12px; }
            QSplitter::handle { background: #2b313c; width: 1px; }
            QToolTip { background: #303949; color: #e2e7ef; border: 1px solid #536077; padding: 6px; }
"""
