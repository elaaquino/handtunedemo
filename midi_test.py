import mido


# Find our HandTune MIDI port.
port_name = None

for name in mido.get_output_names():
    if "HandTune MIDI" in name:
        port_name = name
        break

if port_name is None:
    print("Could not find HandTune MIDI.")
    exit()

print(f"Connected to: {port_name}")


# Open the MIDI output.
with mido.open_output(port_name) as midi_port:

    while True:
        user_input = input(
            "Enter autotune strength (0-100), or q to quit: "
        )

        if user_input.lower() == "q":
            break

        try:
            percentage = int(user_input)

            # Keep the value between 0 and 100.
            percentage = max(0, min(100, percentage))

            # Convert 0-100 into MIDI's 0-127 range.
            midi_value = round(
                (percentage / 100) * 127
            )

            message = mido.Message(
                "control_change",
                channel=0,
                control=20,
                value=midi_value
            )

            midi_port.send(message)

            print(
                f"Sent {percentage}% "
                f"(MIDI value {midi_value})"
            )

        except ValueError:
            print("Please enter a number from 0-100.")