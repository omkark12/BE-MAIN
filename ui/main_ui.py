from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QStackedWidget, QFrame, QGraphicsDropShadowEffect,
    QLineEdit, QSpinBox, QListWidget, QMessageBox, QSizePolicy, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QColor
import threading
from audio_processing.audio_io import AudioProcessor
from audio_processing.reminder import ReminderSystem
from datetime import datetime

class NoiseCancelUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hearing Aid Dashboard")
        self.setGeometry(200, 100, 900, 550)
        self.setStyleSheet("font-family: 'Segoe UI';")
        
        # Initialize reminder system
        self.reminder_system = ReminderSystem()
        self.reminder_system.start()
        
        main_layout = QHBoxLayout()
        sidebar = self.create_sidebar()
        self.stack = QStackedWidget()

        self.home_page = self.create_home_page()
        self.realtime_page = self.create_realtime_page()
        self.reminder_page = self.create_reminder_page()
        self.info_viewer_page = self.create_sliding_info_widget()
        self.about_page = self.create_about_page()

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.realtime_page)
        self.stack.addWidget(self.reminder_page)
        self.stack.addWidget(self.info_viewer_page)
        self.stack.addWidget(self.about_page)

        main_layout.addLayout(sidebar, 1)
        main_layout.addWidget(self.stack, 4)

        container = QWidget()
        container.setLayout(main_layout)
        container.setStyleSheet("background-color: #101820;")
        self.setCentralWidget(container)
        
        # Set up timer to update reminder list
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_reminder_list)
        self.update_timer.start(1000)  # Update every second

    def create_sidebar(self):
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(20, 40, 20, 20)

        buttons = [
            ("🏠 Home", lambda: self.stack.setCurrentWidget(self.home_page)),
            ("🎧 Real-Time", lambda: self.stack.setCurrentWidget(self.realtime_page)),
            ("⏰ Reminders", lambda: self.stack.setCurrentWidget(self.reminder_page)),
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

        # Status indicators
        status_card = QFrame()
        status_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255,255,255,0.05);
                border-radius: 20px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        status_layout = QVBoxLayout()

        # VAD Status
        vad_layout = QHBoxLayout()
        vad_label = QLabel("Voice Activity:")
        vad_label.setStyleSheet("color: #f1f1f1;")
        self.vad_status = QLabel("Inactive")
        self.vad_status.setStyleSheet("color: #d63031; font-weight: bold;")
        vad_layout.addWidget(vad_label)
        vad_layout.addWidget(self.vad_status)
        vad_layout.addStretch()

        # Noise Type
        noise_layout = QHBoxLayout()
        noise_label = QLabel("Noise Type:")
        noise_label.setStyleSheet("color: #f1f1f1;")
        self.noise_type = QLabel("Unknown")
        self.noise_type.setStyleSheet("color: #f1f1f1; font-weight: bold;")
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_type)
        noise_layout.addStretch()

        # Confidence Bar
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("color: #f1f1f1;")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 5px;
                text-align: center;
                background-color: #2d3436;
            }
            QProgressBar::chunk {
                background-color: #00b894;
                border-radius: 5px;
            }
        """)
        self.confidence_bar.setMaximum(100)
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.confidence_bar)

        # Processing Status
        proc_layout = QHBoxLayout()
        proc_label = QLabel("Processing:")
        proc_label.setStyleSheet("color: #f1f1f1;")
        self.proc_status = QLabel("Idle")
        self.proc_status.setStyleSheet("color: #f1f1f1; font-weight: bold;")
        proc_layout.addWidget(proc_label)
        proc_layout.addWidget(self.proc_status)
        proc_layout.addStretch()

        # Add all status indicators
        status_layout.addLayout(vad_layout)
        status_layout.addLayout(noise_layout)
        status_layout.addLayout(conf_layout)
        status_layout.addLayout(proc_layout)
        status_card.setLayout(status_layout)

        # Main control card
        control_card = QFrame()
        control_card.setStyleSheet("""
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
        control_card.setGraphicsEffect(shadow)

        # Initialize audio processor with error handling
        try:
            self.audio_processor = AudioProcessor()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize audio: {str(e)}")
            self.audio_processor = None

        self.start_button = QPushButton("▶ Start Listening")
        self.stop_button = QPushButton("■ Stop Listening")
        self.stop_button.setEnabled(False)

        self.start_button.setStyleSheet(self.button_style("#00b894"))
        self.stop_button.setStyleSheet(self.button_style("#d63031"))

        # Connect buttons with error handling
        self.start_button.clicked.connect(self.start_audio)
        self.stop_button.clicked.connect(self.stop_audio)

        # Noise reduction slider
        slider_layout = QVBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.setStyleSheet(self.slider_style())
        
        # Add value label
        self.slider_value = QLabel("50%")
        self.slider_value.setStyleSheet("color: #f1f1f1; font-size: 14px;")
        self.slider_value.setAlignment(Qt.AlignCenter)
        self.slider.valueChanged.connect(lambda v: self.slider_value.setText(f"{v}%"))

        slider_label = QLabel("Noise Reduction Level")
        slider_label.setStyleSheet("color: #f1f1f1;")
        slider_label.setAlignment(Qt.AlignCenter)

        slider_layout.addWidget(slider_label)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slider_value)

        # Button layout
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.start_button)
        btn_layout.addWidget(self.stop_button)

        # Control layout
        control_layout = QVBoxLayout()
        control_layout.addLayout(btn_layout)
        control_layout.addLayout(slider_layout)
        control_card.setLayout(control_layout)

        # Add cards to main layout
        layout.addWidget(status_card)
        layout.addWidget(control_card)
        layout.addStretch()

        # Initialize status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_audio_status)
        self.status_timer.setInterval(100)  # Update every 100ms

        page.setLayout(layout)
        return page

    def create_reminder_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(20)

        # Create input section
        input_card = QFrame()
        input_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255,255,255,0.05);
                border-radius: 20px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        
        input_layout = QVBoxLayout()
        
        # Task name input
        task_label = QLabel("Task Name:")
        task_label.setStyleSheet("color: #f1f1f1; font-weight: bold;")
        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("e.g., Drink Water, Take a Break, Exercise")
        self.task_input.setStyleSheet("""
            QLineEdit {
                padding: 8px;
                border-radius: 5px;
                background-color: rgba(255,255,255,0.1);
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
                font-size: 14px;
            }
            QLineEdit:focus {
                border: 1px solid rgba(255,255,255,0.4);
                background-color: rgba(255,255,255,0.15);
            }
        """)
        
        # Interval input
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Interval (minutes):")
        interval_label.setStyleSheet("color: #f1f1f1; font-weight: bold;")
        self.interval_input = QSpinBox()
        self.interval_input.setRange(1, 1440)  # 1 minute to 24 hours
        self.interval_input.setValue(30)
        self.interval_input.setStyleSheet("""
            QSpinBox {
                padding: 8px;
                border-radius: 5px;
                background-color: rgba(255,255,255,0.1);
                color: white;
                border: 1px solid rgba(255,255,255,0.2);
                font-size: 14px;
                min-width: 80px;
            }
            QSpinBox:focus {
                border: 1px solid rgba(255,255,255,0.4);
                background-color: rgba(255,255,255,0.15);
            }
            QSpinBox::up-button, QSpinBox::down-button {
                border: none;
                background: rgba(255,255,255,0.1);
                border-radius: 2px;
                margin: 1px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: rgba(255,255,255,0.2);
            }
        """)
        interval_layout.addWidget(interval_label)
        interval_layout.addWidget(self.interval_input)
        interval_layout.addStretch()
        
        # Add button
        add_button = QPushButton("➕ Add Reminder")
        add_button.setStyleSheet(self.button_style("#00b894"))
        add_button.clicked.connect(self.add_reminder)
        
        input_layout.addWidget(task_label)
        input_layout.addWidget(self.task_input)
        input_layout.addLayout(interval_layout)
        input_layout.addWidget(add_button)
        input_card.setLayout(input_layout)
        
        # Create reminder list
        list_card = QFrame()
        list_card.setStyleSheet("""
            QFrame {
                background-color: rgba(255,255,255,0.05);
                border-radius: 20px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }
        """)
        
        list_layout = QVBoxLayout()
        list_label = QLabel("Active Reminders:")
        list_label.setStyleSheet("color: #f1f1f1; font-weight: bold; font-size: 16px;")
        self.reminder_list = QListWidget()
        self.reminder_list.setStyleSheet("""
            QListWidget {
                background-color: rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 10px;
                color: white;
                font-size: 14px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid rgba(255,255,255,0.1);
                margin-bottom: 5px;
            }
            QListWidget::item:selected {
                background-color: rgba(255,255,255,0.2);
                border-radius: 5px;
            }
            QListWidget::item:hover {
                background-color: rgba(255,255,255,0.15);
                border-radius: 5px;
            }
        """)
        
        button_layout = QHBoxLayout()
        
        toggle_button = QPushButton("🔄 Toggle Active/Inactive")
        toggle_button.setStyleSheet(self.button_style("#f1c40f"))
        toggle_button.clicked.connect(self.toggle_reminder)
        
        remove_button = QPushButton("🗑️ Remove Selected")
        remove_button.setStyleSheet(self.button_style("#d63031"))
        remove_button.clicked.connect(self.remove_reminder)
        
        button_layout.addWidget(toggle_button)
        button_layout.addWidget(remove_button)
        
        list_layout.addWidget(list_label)
        list_layout.addWidget(self.reminder_list)
        list_layout.addLayout(button_layout)
        list_card.setLayout(list_layout)
        
        # Connect reminder system signals
        self.reminder_system.reminder_triggered.connect(self.on_reminder_triggered)
        self.reminder_system.reminder_updated.connect(self.update_reminder_list)
        
        layout.addWidget(input_card)
        layout.addWidget(list_card)
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
        """Start audio processing with error handling"""
        if not self.audio_processor:
            QMessageBox.critical(self, "Error", "Audio processor not initialized")
            return

        try:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.proc_status.setText("Starting...")
            self.proc_status.setStyleSheet("color: #fdcb6e; font-weight: bold;")
            
            # Start in a separate thread to avoid UI blocking
            def start_processing():
                try:
                    self.audio_processor.start_stream(self.slider.value())
                except Exception as e:
                    self.handle_audio_error(str(e))
                    return
                
                # Update UI from main thread
                self.proc_status.setText("Active")
                self.proc_status.setStyleSheet("color: #00b894; font-weight: bold;")
                self.status_timer.start()
            
            threading.Thread(target=start_processing, daemon=True).start()
            
        except Exception as e:
            self.handle_audio_error(str(e))

    def stop_audio(self):
        """Stop audio processing with error handling"""
        try:
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.proc_status.setText("Stopping...")
            self.proc_status.setStyleSheet("color: #fdcb6e; font-weight: bold;")
            
            def stop_processing():
                try:
                    self.audio_processor.stop_stream()
                except Exception as e:
                    self.handle_audio_error(str(e))
                    return
                
                # Update UI from main thread
                self.status_timer.stop()
                self.vad_status.setText("Inactive")
                self.vad_status.setStyleSheet("color: #d63031; font-weight: bold;")
                self.noise_type.setText("Unknown")
                self.confidence_bar.setValue(0)
                self.proc_status.setText("Idle")
                self.proc_status.setStyleSheet("color: #f1f1f1; font-weight: bold;")
            
            threading.Thread(target=stop_processing, daemon=True).start()
            
        except Exception as e:
            self.handle_audio_error(str(e))

    def handle_audio_error(self, error_msg):
        """Centralized error handling for audio processing"""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_timer.stop()
        self.proc_status.setText("Error")
        self.proc_status.setStyleSheet("color: #d63031; font-weight: bold;")
        self.vad_status.setText("Inactive")
        self.vad_status.setStyleSheet("color: #d63031; font-weight: bold;")
        self.noise_type.setText("Unknown")
        self.confidence_bar.setValue(0)
        QMessageBox.critical(self, "Error", f"Audio processing error: {error_msg}")

    def update_audio_status(self):
        """Update audio processing status with error handling"""
        if not self.audio_processor or not self.audio_processor.is_active():
            return

        try:
            # Update VAD status
            if self.audio_processor.is_voice_active():
                self.vad_status.setText("Active")
                self.vad_status.setStyleSheet("color: #00b894; font-weight: bold;")
            else:
                self.vad_status.setText("Inactive")
                self.vad_status.setStyleSheet("color: #d63031; font-weight: bold;")

            # Update noise classification
            noise_info = self.audio_processor.get_noise_info()
            if noise_info:
                self.noise_type.setText(noise_info['noise_type'].replace('_', ' ').title())
                self.confidence_bar.setValue(int(noise_info['confidence'] * 100))
            else:
                self.noise_type.setText("Unknown")
                self.confidence_bar.setValue(0)

        except Exception as e:
            self.handle_audio_error(str(e))

    def add_reminder(self):
        task_name = self.task_input.text().strip()
        interval = self.interval_input.value()
        
        if not task_name:
            QMessageBox.warning(self, "Error", "Please enter a task name")
            return
            
        self.reminder_system.add_reminder(task_name, interval)
        self.task_input.clear()
        
    def remove_reminder(self):
        current_item = self.reminder_list.currentItem()
        if current_item:
            task_name = current_item.text().split(" - ")[0].strip()
            self.reminder_system.remove_reminder(task_name)
            
    def toggle_reminder(self):
        current_item = self.reminder_list.currentItem()
        if current_item:
            task_name = current_item.text().split(" - ")[0].strip()
            self.reminder_system.toggle_reminder(task_name)
            
    def on_reminder_triggered(self, task_name):
        """Handle reminder trigger event"""
        # You could add visual feedback here if desired
        pass
            
    def update_reminder_list(self):
        self.reminder_list.clear()
        for reminder in self.reminder_system.get_reminders():
            next_trigger = reminder['next_trigger']
            time_until = next_trigger - datetime.now()
            minutes = max(0, int(time_until.total_seconds() / 60))
            
            status = "🟢 Active" if reminder['active'] else "⚫ Inactive"
            item_text = f"{reminder['task_name']} - {status} - Next: {minutes} minutes"
            self.reminder_list.addItem(item_text)

    def closeEvent(self, event):
        """Handle application close"""
        self.reminder_system.stop()
        super().closeEvent(event)
