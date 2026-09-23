import sys

import cv2
import mediapipe as mp
import math
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar
)

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Palm
    (0, 17)
]

class HandTuneWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("HandTune")
        self.resize(1100, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main page layout
        main_layout = QVBoxLayout(central_widget)

        # -------------------------
        # Header
        # -------------------------

        header_layout = QHBoxLayout()

        title_section = QVBoxLayout()

        title = QLabel("HANDTUNE")
        subtitle = QLabel("Gesture-Controlled Live Audio")

        title_section.addWidget(title)
        title_section.addWidget(subtitle)

        status = QLabel("● READY")

        header_layout.addLayout(title_section)
        header_layout.addStretch()
        header_layout.addWidget(status)

        main_layout.addLayout(header_layout)

        # -------------------------
        # Main content
        # -------------------------

        content_layout = QHBoxLayout()

        # Camera side
        camera_section = QVBoxLayout()

        self.camera_label = QLabel("CAMERA FEED")
        self.camera_label.setMinimumSize(650, 450)
        self.camera_label.setAlignment(Qt.AlignCenter)

        camera_section.addWidget(self.camera_label)

        # Controls / music side
        controls_section = QVBoxLayout()

        now_playing_label = QLabel("NOW PLAYING")
        self.song_name = QLabel("No song selected")

        self.choose_song_button = QPushButton("Choose Song")

        controls_section.addWidget(now_playing_label)
        controls_section.addWidget(self.song_name)
        controls_section.addWidget(self.choose_song_button)

        controls_section.addSpacing(40)

        # Instrumental control
        instrumental_label = QLabel("INSTRUMENTAL")

        self.instrumental_bar = QProgressBar()
        self.instrumental_bar.setRange(0, 100)
        self.instrumental_bar.setValue(0)

        controls_section.addWidget(instrumental_label)
        controls_section.addWidget(self.instrumental_bar)

        controls_section.addSpacing(30)

        # Autotune control
        autotune_label = QLabel("AUTOTUNE")

        self.autotune_bar = QProgressBar()
        self.autotune_bar.setRange(0, 100)
        self.autotune_bar.setValue(0)

        controls_section.addWidget(autotune_label)
        controls_section.addWidget(self.autotune_bar)

        controls_section.addStretch()

        # Put left and right sides together
        content_layout.addLayout(camera_section, 2)
        content_layout.addLayout(controls_section, 1)

        main_layout.addLayout(content_layout)

        # -------------------------
        # Camera
        # -------------------------

        self.camera = cv2.VideoCapture(0)

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(30)
    
    def update_camera(self):
        success, frame = self.camera.read()

        if not success:
            return

        # OpenCV uses BGR.
        # Qt expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channels = rgb_frame.shape

        bytes_per_line = channels * width

        qt_image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)

        pixmap = pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.camera_label.setPixmap(pixmap)

        def closeEvent(self, event):
            self.camera_timer.stop()

            if self.camera.isOpened():
                self.camera.release()

            event.accept()


def main():
    app = QApplication(sys.argv)

    window = HandTuneWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()