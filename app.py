import sys
from unittest import result

import cv2
import mediapipe as mp
import math
import time
import mido

import soundfile as sf
import sounddevice as sd
import numpy as np

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
    QProgressBar,
    QFileDialog
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

        self.choose_song_button.clicked.connect(
            self.choose_song
        )

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

        # -------------------------
        # Audio
        # -------------------------

        self.audio_data = None
        self.sample_rate = None
        self.audio_position = 0

        self.instrumental_volume = 0.0

        self.audio_stream = None

        # -------------------------
        # MIDI
        # -------------------------

        self.midi_port = None
        self.last_midi_value = None

        for name in mido.get_output_names():
            if "HandTune MIDI" in name:
                self.midi_port = mido.open_output(name)
                print(f"Connected to MIDI: {name}")
                break

        if self.midi_port is None:
            print("HandTune MIDI not found.")
        
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

            openness = calculate_hand_openness(hand_landmarks)

            if hand_label not in self.smoothed_openness:
                self.smoothed_openness[hand_label] = openness

            smoothing = 0.15

            self.smoothed_openness[hand_label] = (
                self.smoothed_openness[hand_label] * (1 - smoothing)
                + openness * smoothing
            )

            smooth_value = self.smoothed_openness[hand_label]

            if hand_label == "Left":
                self.instrumental_volume = smooth_value * 0.55

                self.instrumental_bar.setValue(
                    round(smooth_value * 100)
                )

            elif hand_label == "Right":
                autotune_percent = round(smooth_value * 100)

                self.autotune_bar.setValue(
                    autotune_percent
                )

                midi_value = round(smooth_value * 127)

                if (
                    self.midi_port is not None
                    and midi_value != self.last_midi_value
                ):
                    message = mido.Message(
                        "control_change",
                        channel=0,
                        control=20,
                        value=midi_value
                    )

                    self.midi_port.send(message)
                    self.last_midi_value = midi_value

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

        current_time = time.time()

        if current_time - self.last_seen["Left"] > 0.4:
            current_value = self.instrumental_bar.value()

            new_value = round(current_value * 0.9)

            if new_value < 1:
                new_value = 0

            self.instrumental_bar.setValue(new_value)

            self.instrumental_volume = (
                new_value / 100
            ) * 0.55


        if current_time - self.last_seen["Right"] > 0.4:
            current_value = self.autotune_bar.value()

            new_value = round(current_value * 0.9)

            if new_value < 1:
                new_value = 0

            self.autotune_bar.setValue(new_value)

            midi_value = round((new_value / 100) * 127)

            if (
                self.midi_port is not None
                and midi_value != self.last_midi_value
            ):
                message = mido.Message(
                    "control_change",
                    channel=0,
                    control=20,
                    value=midi_value
                )

                self.midi_port.send(message)
                self.last_midi_value = midi_value

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

    def choose_song(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Instrumental",
            "",
            "Audio Files (*.wav)"
        )

        if not file_path:
            return

        try:
            # Stop previous song if necessary
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None

            # Load selected song
            self.audio_data, self.sample_rate = sf.read(
                file_path,
                dtype="float32",
                always_2d=True
            )

            self.audio_position = 0

            # Display filename
            file_name = file_path.split("/")[-1]
            self.song_name.setText(file_name)

            print(f"Loaded song: {file_name}")

            self.start_audio()

        except Exception as error:
            print(f"Could not load song: {error}")

    def start_audio(self):
            if self.audio_data is None:
                return
    
            channels = self.audio_data.shape[1]
    
            self.audio_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=channels,
                dtype="float32",
                callback=self.audio_callback
            )
    
            self.audio_stream.start()

    def audio_callback(self, outdata, frames, time_info, status):
        if status:
            print(status)

        if self.audio_data is None:
            outdata.fill(0)
            return

        start = self.audio_position
        end = start + frames

        chunk = self.audio_data[start:end]

        # Clear output first
        outdata.fill(0)

        available_frames = len(chunk)

        if available_frames > 0:
            outdata[:available_frames] = (
                chunk * self.instrumental_volume
            )

        self.audio_position += available_frames

        # Song finished
        if self.audio_position >= len(self.audio_data):
            self.audio_position = len(self.audio_data)

    def closeEvent(self, event):
        self.camera_timer.stop()

        if self.camera.isOpened():
            self.camera.release()

        self.landmarker.close()

        if self.midi_port is not None:
            self.midi_port.close()

        if self.audio_stream is not None:
            self.audio_stream.stop()
            self.audio_stream.close()

        event.accept()


def main():
    app = QApplication(sys.argv)

    window = HandTuneWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()