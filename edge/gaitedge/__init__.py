"""gaitedge: the headless device side of gaitscreen.

Runs on a small computer with a camera attached (built for a Raspberry Pi 5),
with no screen and no browser. It watches the camera, records each walk that
passes, and then either analyses the walk on the device or sends the recording
to a server to be analysed there, depending on what the device can keep up
with (see ``gaitedge bench`` and the README).

All gait analysis comes from the ``gaitscreen`` package in this repository,
installed as a dependency rather than copied, so the device and the server can
never disagree about how a walk is measured.
"""

__version__ = "0.1.0"
