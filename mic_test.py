import sounddevice as sd


INPUT_DEVICE = 1
OUTPUT_DEVICE = 5

SAMPLE_RATE = 48000
CHANNELS = 1


def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)

    outdata[:] = indata


with sd.Stream(
    device=(INPUT_DEVICE, OUTPUT_DEVICE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    callback=audio_callback
):
    print("Microphone monitoring active.")
    print("Press Enter to stop.")

    input()