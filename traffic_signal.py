from configurations import SIGNAL_CYCLE_SECONDS

def is_red_light(frame_number, fps):
    seconds = frame_number // fps
    return (seconds // SIGNAL_CYCLE_SECONDS) % 2 == 0
