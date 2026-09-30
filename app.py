import sys
from unittest import result

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

from main import CLOSED_HAND_VALUE
from main import OPEN_HAND_VALUE

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

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

def calculate_hand_openness(hand_landmarks):
    wrist = hand_landmarks[0]
    index_mcp = hand_landmarks[5]
    middle_mcp = hand_landmarks[9]
    pinky_mcp = hand_landmarks[17]

    fingertip_indices = [8, 12, 16, 20]

    palm_length = distance(wrist, middle_mcp)
    palm_width = distance(index_mcp, pinky_mcp)

    palm_size = (palm_length + palm_width) / 2

    distances = []

    for tip_index in fingertip_indices:
        fingertip = hand_landmarks[tip_index]

        fingertip_distance = distance(wrist, fingertip)

        # Normalize the fingertip distance based on hand size.
        normalized_distance = fingertip_distance / palm_size

        distances.append(normalized_distance)

    average_distance = sum(distances) / len(distances)

    openness = (
        (average_distance - CLOSED_HAND_VALUE)
        / (OPEN_HAND_VALUE - CLOSED_HAND_VALUE)
    )

    openness = max(0.0, min(1.0, openness))

    return openness

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

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path="models/hand_landmarker.task"
            ),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        self.frame_timestamp = 0

        self.smoothed_openness = {}

        self.last_seen = {
            "Left": time.time(),
            "Right": time.time()
        }

        self.camera = cv2.VideoCapture(0)

        self.debug_frame_count = 0

        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera)
        self.camera_timer.start(30)
    
    def update_camera(self):
        success, frame = self.camera.read()

        if not success:
            return

        # Convert camera frame for MediaPipe
        rgb_for_mediapipe = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_for_mediapipe
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            self.frame_timestamp
        )

        self.frame_timestamp += 33

        height, width, _ = frame.shape

        # Process and draw any detected hands
        for hand_index, hand_landmarks in enumerate(
            result.hand_landmarks
        ):
            handedness = result.handedness[hand_index][0]
            hand_label = handedness.category_name

            self.last_seen[hand_label] = time.time()

            # Draw connections
            for start_index, end_index in HAND_CONNECTIONS:
                start = hand_landmarks[start_index]
                end = hand_landmarks[end_index]

                start_point = (
                    int(start.x * width),
                    int(start.y * height)
                )

                end_point = (
                    int(end.x * width),
                    int(end.y * height)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (255, 255, 255),
                    2
                )

            # Draw landmarks
            for landmark in hand_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

        # -------------------------
        # Display frame in Qt
        # -------------------------
        # IMPORTANT: this is OUTSIDE the hand loop.
        # The camera therefore updates even when no hands are detected.

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

        self.landmarker.close()

        event.accept()


def main():
    app = QApplication(sys.argv)

    window = HandTuneWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()