from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QStackedWidget, QFrame, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import threading
from audio_processing.audio_io import AudioProcessor
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QPushButton, QStackedWidget, QHBoxLayout
from PyQt5.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve
from PyQt5.QtWidgets import QSizePolicy, QGraphicsDropShadowEffect
from PyQt5.QtGui import QColor
from utils.visualization import MFFCCanvas




class NoiseCancelUI(QMainWindow):
    def update_mfcc_plot(self, mfcc):
        if hasattr(self, "mfcc_canvas") and mfcc is not None:
            self.mfcc_canvas.plot_mfcc(mfcc, 44100)


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hearing Aid Dashboard")
        self.setGeometry(200, 100, 900, 550)
        self.setStyleSheet("font-family: 'Segoe UI';")

        main_layout = QHBoxLayout()
        sidebar = self.create_sidebar()
        self.stack = QStackedWidget()

        self.home_page = self.create_home_page()
        self.realtime_page = self.create_realtime_page()
        self.info_viewer_page = self.create_sliding_info_widget()
        self.about_page = self.create_about_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.realtime_page)
        self.stack.addWidget(self.info_viewer_page)
        self.stack.addWidget(self.about_page)

        main_layout.addLayout(sidebar, 1)
        main_layout.addWidget(self.stack, 4)

        container = QWidget()
        container.setLayout(main_layout)
        container.setStyleSheet("background-color: #101820;")
        self.setCentralWidget(container)

    def create_sidebar(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 40, 20, 20)

        buttons = [
            ("🏠 Home", lambda: self.stack.setCurrentWidget(self.home_page)),
            ("🎧 Real-Time", lambda: self.stack.setCurrentWidget(self.realtime_page)),
            ("🧾 Tech-Stack", lambda: self.stack.setCurrentWidget(self.info_viewer_page)),
            ("ℹ️ About", lambda: self.stack.setCurrentWidget(self.about_page)),
        ]

        for text, action in buttons:
            btn = QPushButton(text)
            btn.setStyleSheet(self.nav_button_style())
            btn.clicked.connect(action)
            layout.addWidget(btn)

        layout.addStretch()
        return layout

    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Title
        title = QLabel("ML POWERED PERSONALISED HEARING AID")
        title.setStyleSheet("color: #f2f2f2; font-size: 26px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)

        # Subtitle
        subtitle = QLabel("Enhancing your listening experience with real-time noise cancellation and custom settings.")
        subtitle.setStyleSheet("color: #bdc3c7; font-size: 16px;")
        subtitle.setAlignment(Qt.AlignCenter)

        # Call to Action
        cta_label = QLabel("Choose an option from the menu to start.")
        cta_label.setStyleSheet("color: #f2f2f2; font-size: 18px;")
        cta_label.setAlignment(Qt.AlignCenter)

        # Add banner or images section here (skip for now)
        # You can add an image or some banner-like text or a background here, as discussed earlier

        # Add widgets to the layout
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(cta_label)

        # Add more info cards (optional)
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setOffset(0, 4)
        card_shadow.setColor(QColor(0, 0, 0, 160))
        card.setGraphicsEffect(card_shadow)

        card_layout = QVBoxLayout()
        card_title = QLabel("Features of the Dashboard")
        card_title.setStyleSheet("color: #f2f2f2; font-size: 20px; font-weight: bold;")
        card_title.setAlignment(Qt.AlignCenter)

        feature1 = QLabel("✔ Real-time noise cancellation")
        feature1.setStyleSheet("color: #bdc3c7; font-size: 16px;")
        feature1.setAlignment(Qt.AlignCenter)

        feature2 = QLabel("✔ Customizable noise reduction levels")
        feature2.setStyleSheet("color: #bdc3c7; font-size: 16px;")
        feature2.setAlignment(Qt.AlignCenter)

        feature3 = QLabel("✔ Seamless user experience")
        feature3.setStyleSheet("color: #bdc3c7; font-size: 16px;")
        feature3.setAlignment(Qt.AlignCenter)

        card_layout.addWidget(card_title)
        card_layout.addWidget(feature1)
        card_layout.addWidget(feature2)
        card_layout.addWidget(feature3)
        card.setLayout(card_layout)

        layout.addWidget(card)
        layout.addStretch()

        page.setLayout(layout)
        return page

    def create_realtime_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: rgba(255,255,255,0.05);
                border-radius: 20px;
                padding: 30px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setOffset(0, 5)
        shadow.setColor(QColor(0, 0, 0, 180))
        card.setGraphicsEffect(shadow)

        self.audio_processor = AudioProcessor()
        self.start_button = QPushButton("▶ Start Listening")
        self.stop_button = QPushButton("■ Stop Listening")
        self.stop_button.setEnabled(False)

        self.start_button.setStyleSheet(self.button_style("#00b894"))
        self.stop_button.setStyleSheet(self.button_style("#d63031"))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setStyleSheet(self.slider_style())

        label = QLabel("Noise Reduction Level")
        label.setStyleSheet("color: #f1f1f1;")
        label.setAlignment(Qt.AlignCenter)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.stop_button)

        card_layout = QVBoxLayout()
        card_layout.addLayout(btn_layout)
        card_layout.addWidget(label)
        card_layout.addWidget(self.slider)
        card.setLayout(card_layout)
        self.mfcc_canvas = MFFCCanvas()
        card_layout.addWidget(self.mfcc_canvas)



        self.start_button.clicked.connect(self.start_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        layout.addWidget(card)
        layout.addStretch()
        page.setLayout(layout)
        return page



    def create_sliding_info_widget(self):
        page = QWidget()
        page.setStyleSheet("background-color: #1e272e;")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(20)

        # Title
        title = QLabel("📚 Technologies & Algorithms Used")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #f2f2f2;")
        layout.addWidget(title)

        # Stacked Widget
        stack = QStackedWidget()
        stack.setStyleSheet("background-color: transparent;")
        layout.addWidget(stack)

        # Cards with shadow effect
        def create_card(title, points):
        # Outer container widget
            container = QWidget()
            container_layout = QHBoxLayout()
            container_layout.setAlignment(Qt.AlignCenter)
            container.setLayout(container_layout)
            # You can adjust this if needed
            # Actual card widget with shadow
            card = QWidget()
            card.setStyleSheet("""
                QWidget {
                    background-color: #2f3640;
                    border-radius: 12px;
                }
            """)
            card.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
            card.setMinimumWidth(500)

            # Add drop shadow effect
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 5)
            shadow.setColor(QColor(0, 0, 0, 160))
            card.setGraphicsEffect(shadow)

            # Card layout and content
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(35, 35, 35, 35)
            card_layout.setSpacing(16)

            # Heading
            heading = QLabel(title)
            heading.setAlignment(Qt.AlignCenter)
            heading.setStyleSheet("font-size: 24px; font-weight: bold; color: #f2f2f2;")
            card_layout.addWidget(heading)

            # Points
            for point in points:
                label = QLabel(f"✔ {point}")
                label.setStyleSheet("font-size: 16px; color: #bdc3c7;")
                label.setAlignment(Qt.AlignCenter)
                card_layout.addWidget(label)

            container_layout.addWidget(card)
            return container


        # Add your algorithm/tech slides
        stack.addWidget(create_card("MVDR Beamforming", [
            "Enhances speech signal from target direction",
            "Suppresses background noise using spatial filtering",
            "Uses microphone array signal covariance matrix"
        ]))

        stack.addWidget(create_card("Voice Activity Detection (VAD)", [
            "Detects when speech is present in audio",
            "Helps in applying enhancement only during speech",
            "Reduces unnecessary processing and power usage"
        ]))

        stack.addWidget(create_card("Machine Learning-based Filtering", [
            "Applies ML to identify noise patterns",
            "Dynamic adjustment of filters in real-time",
            "Improves clarity even in non-stationary noise"
        ]))

        stack.addWidget(create_card("Libraries", [
            "Python for backend + real-time logic",
            "PyQt5 for modern UI",
            "NumPy and SciPy for signal processing",
            "Real-time audio capture & processing"
        ]))

        # Controls
        controls = QHBoxLayout()
        prev_btn = QPushButton("⬅️ Previous")
        next_btn = QPushButton("Next ➡️")
        
        for btn in (prev_btn, next_btn):
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #101820;
                    color: #2d3436;
                    font-weight: bold;
                    padding: 12px 20px;
                    border-radius: 12px;
                    font-size: 18px;
                }
                QPushButton:hover {
                    background-color: #bdc3c7;
                    color: #2d3436;
                }
            """)

        controls.addWidget(prev_btn)
        controls.addStretch()
        controls.addWidget(next_btn)
        layout.addLayout(controls)

        # Animation Function
        def slide_to(index):
            curr_index = stack.currentIndex()
            if index == curr_index or index < 0 or index >= stack.count():
                return

            current_widget = stack.currentWidget()
            next_widget = stack.widget(index)

            direction = -1 if index > curr_index else 1
            w = stack.width()

            next_widget.setGeometry(QRect(direction * w, 0, w, stack.height()))
            stack.setCurrentIndex(index)

            anim = QPropertyAnimation(next_widget, b"geometry")
            anim.setDuration(300)
            anim.setStartValue(QRect(direction * w, 0, w, stack.height()))
            anim.setEndValue(QRect(0, 0, w, stack.height()))
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.start()

            # Hold ref to prevent garbage collection
            stack.animation = anim

        # Button Logic
        prev_btn.clicked.connect(lambda: slide_to(stack.currentIndex() - 1))
        next_btn.clicked.connect(lambda: slide_to(stack.currentIndex() + 1))

        return page



    def create_about_page(self):
        about_page = QWidget()
        about_page.setStyleSheet("background-color: #1e272e;")
        layout = QVBoxLayout()
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setSpacing(25)

        def create_label(text, font_size=16, bold=False, color="#dfe6e9", top_padding=0):
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setWordWrap(True)
            label.setStyleSheet(f"""
                font-size: {font_size}px;
                {'font-weight: bold;' if bold else ''}
                color: {color};
                padding-top: {top_padding}px;
            """)
            return label

        # Title
        layout.addWidget(create_label("🔊 ML POWERED PERSONALIZED HEARING AID", 30, True, "#f2f2f2"))

        # Subtitle
        layout.addWidget(create_label("Final Year B.E. (Computer Engineering) Project", 16, False))

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("color: #636e72;")
        layout.addWidget(divider)

        # Developed By
        layout.addWidget(create_label("👨‍💻 Developed By", 20, True, "#f2f2f2"))

        dev_names = QLabel("• Nishant Gangurde\n• Omkar Katkar\n• Aditya Kulkarni\n• Khushi Raval")
        dev_names.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dev_names.setStyleSheet("""
            font-size: 16px;
            color: #dfe6e9;
            line-height: 26px;
        """)
        layout.addWidget(dev_names)

        # Project Goal
        layout.addWidget(create_label("🎯 Project Objective", 20, True, "#f2f2f2", 10))

        goal = QLabel(
            "A next-gen hearing aid platform that enhances speech and eliminates background noise using:\n"
            "• MVDR Beamforming\n• Voice Activity Detection (VAD)\n• Real-Time Machine Learning-based Audio Processing"
        )
        goal.setWordWrap(True)
        goal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        goal.setStyleSheet("""
            font-size: 16px;
            color: #dfe6e9;
            line-height: 26px;
        """)
        layout.addWidget(goal)

        # Tech Stack
        layout.addWidget(create_label("🛠️ Tech Stack: Python • PyQt5 • NumPy • Real-Time Audio", 15, False, "#a4b0be", 10))

        # Version & Contact
        layout.addWidget(create_label("📦 Version 1.0.0       📅 April 2025", 14, False, "#a4b0be"))

        contact = QLabel("📧 Email: <a href='mailto:agkulkarni2003@gmail.com'>agkulkarni2003@gmail.com</a>")
        contact.setOpenExternalLinks(True)
        contact.setAlignment(Qt.AlignmentFlag.AlignCenter)
        contact.setStyleSheet("font-size: 14px; color: #74b9ff;")
        layout.addWidget(contact)

        # GitHub link
       # github = QLabel('<a href="https://github.com/yourprojectlink">🌐 View GitHub Repository</a>')
        #github.setOpenExternalLinks(True)
        #github.setAlignment(Qt.AlignmentFlag.AlignCenter)
        #github.setStyleSheet("font-size: 14px; color: #74b9ff;")
        #layout.addWidget(github)

        layout.addStretch()
        about_page.setLayout(layout)
        return about_page



    def button_style(self, color):
        return f"""
        QPushButton {{
            background-color: {color};
            color: white;
            font-size: 14px;
            padding: 12px 20px;
            border-radius: 10px;
        }}
        QPushButton:hover {{
            background-color: {color[:-1]}d;
        }}
        """

    def nav_button_style(self):
        return """
        QPushButton {
            background-color: rgba(255,255,255,0.07);
            color: #ecf0f1;
            padding: 10px;
            border-radius: 10px;
            font-size: 14px;
            text-align: left;
        }
        QPushButton:hover {
            background-color: rgba(255,255,255,0.15);
        }
        """

    def slider_style(self):
        return """
        QSlider::groove:horizontal {
            height: 8px;
            background: #2c3e50;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #00cec9;
            width: 20px;
            margin: -6px 0;
            border-radius: 10px;
        }
        """
        
    def start_audio(self):
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        threading.Thread(
            target=self.audio_processor.start_processing,
            args=(self.slider.value(), self.update_mfcc_plot),  # ✅ Pass callback properly
            daemon=True
        ).start()



    def stop_audio(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.audio_processor.stop_processing()
