import sounddevice as sd
import soundfile as sf

MAX_VOLUME = 0.55

# Load the instrumental.
audio_data, sample_rate = sf.read(
    "audio/Kanye_West_-_Heartless_Instrumental.wav"
)

# Keep track of where we are in the song.
current_frame = 0

# Volume ranges from 0.0 to 1.0.
volume = 1.0


def audio_callback(outdata, frames, time, status):
    global current_frame

    if status:
        print(status)

    # Figure out which part of the song we need.
    end_frame = current_frame + frames

    # Get that chunk of audio.
    chunk = audio_data[current_frame:end_frame]

    # If we reach the end of the song, fill the remaining
    # output with silence.
    if len(chunk) < frames:
        outdata[:] = 0
        outdata[:len(chunk)] = chunk * volume * MAX_VOLUME
        raise sd.CallbackStop()

    # Apply our volume to the audio.
    outdata[:] = chunk * volume * MAX_VOLUME

    # Move forward through the song.
    current_frame = end_frame


# Determine whether the file is mono or stereo.
if len(audio_data.shape) == 1:
    channels = 1
else:
    channels = audio_data.shape[1]


with sd.OutputStream(
    samplerate=sample_rate,
    channels=channels,
    callback=audio_callback
):
    while True:
        user_input = input("Volume 0-100, or q to quit: ")

        if user_input == "q":
            break

        volume = int(user_input) / 100