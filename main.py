import cv2
import mediapipe as mp
import math

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),

    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),

    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),

    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),

    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),

    # Bottom of palm
    (0, 17)
]

CLOSED_HAND_VALUE = 0.95
OPEN_HAND_VALUE = 2.35

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

def main():
    # Create the MediaPipe Hand Landmarker
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

    landmarker = HandLandmarker.create_from_options(options)

    # Open webcam
    camera = cv2.VideoCapture(0)

    frame_timestamp = 0

    smoothed_openness = {}

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read camera.")
            break

        # OpenCV uses BGR colors.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the OpenCV image into a MediaPipe image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Detect hands.
        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )

        frame_timestamp += 33

        # Get camera dimensions.
        height, width, _ = frame.shape

        # Draw every detected landmark.
        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):

            # Calculate the current raw openness.
            openness = calculate_hand_openness(hand_landmarks)

            # If this is the first time seeing this hand,
            # start its smoothed value at the current openness.
            if hand_index not in smoothed_openness:
                smoothed_openness[hand_index] = openness

            # Controls how quickly the value responds.
            smoothing = 0.15

            # Smooth the current value with the previous value.
            smoothed_openness[hand_index] = (
                smoothed_openness[hand_index] * (1 - smoothing)
                + openness * smoothing
            )

            smooth_value = smoothed_openness[hand_index]

            # Convert 0.0 - 1.0 into 0 - 100%.
            percentage = round(smooth_value * 100)

            print(f"Hand {hand_index}: {percentage}%")

            # Determine left or right hand.
            handedness = result.handedness[hand_index][0]
            hand_label = handedness.category_name

            # Get wrist position
            wrist = hand_landmarks[0]
            wrist_x = int(wrist.x * width)
            wrist_y = int(wrist.y * height)

            # Draw connections between landmarks.
            for start_index, end_index in HAND_CONNECTIONS:
                start = hand_landmarks[start_index]
                end = hand_landmarks[end_index]

                start_x = int(start.x * width)
                start_y = int(start.y * height)

                end_x = int(end.x * width)
                end_y = int(end.y * height)

                cv2.line(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
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

        cv2.imshow("HandTune", frame)

        # Press Q to quit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()