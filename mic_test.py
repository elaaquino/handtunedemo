import sounddevice as sd
import aubio
import numpy as np

INPUT_DEVICE = 1
OUTPUT_DEVICE = 5

SAMPLE_RATE = 48000
CHANNELS = 1

BUFFER_SIZE = 2048
HOP_SIZE = 512

# Create the pitch detector.
pitch_detector = aubio.pitch(
    "yin",
    BUFFER_SIZE,
    HOP_SIZE,
    SAMPLE_RATE
)

# Tell aubio to return pitch in Hz.
pitch_detector.set_unit("Hz")

# Ignore very uncertain pitch estimates.
pitch_detector.set_tolerance(0.8)

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]


def frequency_to_note(frequency):
    midi_note = round(
        69 + 12 * np.log2(frequency / 440.0)
    )

    note_name = NOTE_NAMES[midi_note % 12]
    octave = (midi_note // 12) - 1

    return note_name, octave, midi_note

def get_pitch_correction(frequency, midi_note):
    # Find the exact frequency of the nearest note.
    target_frequency = 440.0 * (2 ** ((midi_note - 69) / 12))

    # Calculate how far the voice is from that note, in cents.
    cents_difference = 1200 * np.log2(
        target_frequency / frequency
    )

    return target_frequency, cents_difference

def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    # Send the microphone directly to the headphones.
    outdata[:] = indata

    # Give the original microphone audio to aubio.
    audio_samples = indata[:, 0].astype(np.float32)

    # Detect the pitch.
    pitch = pitch_detector(audio_samples)[0]

    # Get aubio's confidence in that pitch.
    confidence = pitch_detector.get_confidence()

    # Only print reasonably confident detections.
    if confidence > 0.8 and pitch > 0:
        note_name, octave, midi_note = frequency_to_note(pitch)

        target_frequency, cents_difference = get_pitch_correction(
            pitch,
            midi_note
        )

        print(
            f"Pitch: {pitch:.1f} Hz | "
            f"Note: {note_name}{octave} | "
            f"Target: {target_frequency:.1f} Hz | "
            f"Correction: {cents_difference:+.1f} cents"
        )


with sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    blocksize=HOP_SIZE,
    callback=audio_callback
):
    print("Pitch detection active.")
    print("Sing into the microphone.")
    print("Press Enter to stop.")

    input()