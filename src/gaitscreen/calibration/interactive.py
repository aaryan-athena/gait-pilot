"""Interactive point picking for the one-time calibration step.

Matplotlib is used rather than an OpenCV window because it behaves consistently
across platforms and gives zoom for free, which matters when the operator is
clicking a floor line accurately.

Returned coordinates are image pixels with y DOWN -- the same convention
:class:`~gaitscreen.calibration.model.Calibration` stores.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def grab_frame(video_path: str | Path, frame_index: int = 0) -> np.ndarray:
    """Read a single RGB frame for the operator to click on."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    try:
        if frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"could not read frame {frame_index} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def pick_points(
    video_path: str | Path,
    n_points: int,
    *,
    frame_index: int = 0,
    instructions: str | None = None,
) -> list[list[float]]:
    """Show a frame and collect ``n_points`` clicked positions.

    Zoom and pan with the toolbar first if needed; only clicks in the image area
    are recorded, and the last click can be undone with backspace.
    """
    import matplotlib.pyplot as plt

    image = grab_frame(video_path, frame_index)
    figure, axes = plt.subplots(figsize=(11, 7))
    axes.imshow(image)
    axes.set_title(
        instructions
        or f"Click {n_points} reference point(s); backspace undoes, close when done"
    )
    axes.set_xlabel("Zoom with the toolbar before clicking for better accuracy")

    picked: list[list[float]] = []
    markers: list = []

    def on_click(event):
        if event.inaxes is not axes or event.xdata is None:
            return
        if len(picked) >= n_points:
            return
        picked.append([float(event.xdata), float(event.ydata)])
        markers.append(axes.plot(event.xdata, event.ydata, "o", ms=9, mfc="none",
                                 mec="red", mew=2)[0])
        markers.append(axes.annotate(str(len(picked)), (event.xdata, event.ydata),
                                     color="red", fontsize=12,
                                     xytext=(8, 8), textcoords="offset points"))
        if len(picked) >= 2:
            xs = [p[0] for p in picked[-2:]]
            ys = [p[1] for p in picked[-2:]]
            markers.append(axes.plot(xs, ys, "-", color="red", lw=1)[0])
        figure.canvas.draw_idle()
        if len(picked) == n_points:
            axes.set_title("All points marked -- close the window to continue")
            figure.canvas.draw_idle()

    def on_key(event):
        if event.key == "backspace" and picked:
            picked.pop()
            while markers:
                artist = markers.pop()
                artist.remove()
                if not isinstance(artist, plt.Line2D) or len(picked) == 0:
                    break
            _redraw()

    def _redraw():
        while markers:
            markers.pop().remove()
        for i, (x, y) in enumerate(picked, start=1):
            markers.append(axes.plot(x, y, "o", ms=9, mfc="none", mec="red", mew=2)[0])
            markers.append(axes.annotate(str(i), (x, y), color="red", fontsize=12,
                                         xytext=(8, 8), textcoords="offset points"))
        axes.set_title(f"Click {n_points - len(picked)} more point(s)")
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("button_press_event", on_click)
    figure.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    if len(picked) != n_points:
        raise ValueError(f"expected {n_points} points, got {len(picked)}")
    return picked
