"""Figures for the app and the session report.

The event-overlay figure is the most important one here. Everything downstream --
stride times, variability, asymmetry, double support -- rests on gait events being
detected in the right places, and that is a judgement a person can make instantly
from a picture and not at all from a table of numbers. In a pilot deployment,
where the failure mode is trusting a plausible-looking wrong number, showing the
working matters more than showing the result.
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..pose.schema import SIDE_LANDMARKS  # noqa: E402
from ..segmentation.events import _anterior  # noqa: E402

SIDE_COLOUR = {"left": "#2c7fb8", "right": "#d95f0e"}
GRID = dict(alpha=0.25, linewidth=0.6)


def _style(axes) -> None:
    axes.grid(True, **GRID)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)


def event_overlay_figure(series, analysis, *, max_seconds: float = 12.0):
    """Heel anterior position per limb, with detected events marked.

    Heel strikes should sit on the peaks and toe-offs in the troughs, and the two
    limbs should alternate. Anything else is visible at a glance.
    """
    figure, axes_list = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True, constrained_layout=True
    )
    direction = analysis.passes[0].direction if analysis.passes else 1
    limit = min(max_seconds, float(series.t[-1])) if series.n_frames else 0.0

    for axes, side in zip(axes_list, ("left", "right")):
        signal = _anterior(series, side, "heel", direction)
        mask = series.t <= limit
        axes.plot(series.t[mask], signal[mask], color=SIDE_COLOUR[side], lw=1.4,
                  label=f"{side} heel (anterior, relative to pelvis)")

        for kind, marker, size in (("heel_strike", "v", 60), ("toe_off", "^", 45)):
            times = [e.t for e in analysis.events
                     if e.side == side and e.kind == kind and e.t <= limit]
            if not times:
                continue
            values = np.interp(times, series.t, signal)
            axes.scatter(times, values, marker=marker, s=size, zorder=3,
                         facecolors="none" if kind == "toe_off" else SIDE_COLOUR[side],
                         edgecolors=SIDE_COLOUR[side], linewidths=1.4,
                         label=kind.replace("_", " "))

        for cycle in analysis.cycles:
            if cycle.side != side or cycle.t_start > limit:
                continue
            axes.axvspan(cycle.t_start, min(cycle.t_end, limit),
                         color="#2ca25f" if cycle.valid else "#bdbdbd",
                         alpha=0.10 if cycle.valid else 0.18, lw=0)

        axes.set_ylabel("pixels")
        axes.legend(loc="upper right", fontsize=8, framealpha=0.9)
        _style(axes)

    axes_list[-1].set_xlabel("time (s)")
    axes_list[0].set_title(
        "Detected gait events — heel strikes should land on peaks, toe-offs in "
        "troughs; shaded bands are strides (grey = excluded)",
        fontsize=10, loc="left",
    )
    return figure


def stride_time_figure(analysis):
    """Stride-to-stride durations, which is what variability is computed from."""
    figure, axes = plt.subplots(figsize=(11, 3.0), constrained_layout=True)
    plotted = False

    for side in ("left", "right"):
        cycles = [c for c in analysis.cycles if c.side == side and c.valid]
        if not cycles:
            continue
        plotted = True
        axes.plot([c.t_start for c in cycles], [c.duration_s for c in cycles],
                  "o-", color=SIDE_COLOUR[side], ms=5, lw=1.2, label=f"{side} strides")

    excluded = [c for c in analysis.cycles if not c.valid]
    if excluded:
        axes.plot([c.t_start for c in excluded], [c.duration_s for c in excluded],
                  "x", color="#999999", ms=7, label="excluded")

    if plotted:
        valid = [c.duration_s for c in analysis.cycles if c.valid]
        mean = float(np.mean(valid))
        axes.axhline(mean, color="#444444", lw=0.9, ls="--",
                     label=f"mean {mean:.2f}s")

    axes.set_xlabel("time (s)")
    axes.set_ylabel("stride time (s)")
    axes.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    _style(axes)
    return figure


def angle_curves_figure(angles):
    """Mean joint angle curves with cycle-to-cycle spread."""
    joints = ("hip", "knee", "ankle")
    figure, axes_list = plt.subplots(
        1, 3, figsize=(11, 3.2), constrained_layout=True
    )

    for axes, joint in zip(axes_list, joints):
        drawn = False
        for side in ("left", "right"):
            rows = angles.curves.get(f"{side}_{joint}")
            if rows is None or rows.size == 0:
                continue
            drawn = True
            mean = np.nanmean(rows, axis=0)
            sd = np.nanstd(rows, axis=0)
            axes.plot(angles.percent, mean, color=SIDE_COLOUR[side], lw=1.6, label=side)
            axes.fill_between(angles.percent, mean - sd, mean + sd,
                              color=SIDE_COLOUR[side], alpha=0.15, lw=0)
        axes.set_title(f"{joint} flexion", fontsize=10)
        axes.set_xlabel("% of gait cycle")
        if not drawn:
            axes.text(0.5, 0.5, "no valid cycles", ha="center", va="center",
                      transform=axes.transAxes, color="#888888", fontsize=9)
        _style(axes)

    axes_list[0].set_ylabel("degrees")
    axes_list[0].legend(loc="upper right", fontsize=8)
    return figure


def trend_figure(
    history,
    metric: str,
    label: str,
    *,
    flagged_dates: Optional[Sequence] = None,
    baseline_centre: Optional[float] = None,
    baseline_spread: Optional[float] = None,
    low_confidence_column: str = "low_confidence",
):
    """One metric over time, with flagged sessions and the personal baseline."""
    figure, axes = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
    frame = history.dropna(subset=[metric]).sort_values("session_date")

    if frame.empty:
        axes.text(0.5, 0.5, f"no {label} measurements yet", ha="center", va="center",
                  transform=axes.transAxes, color="#888888")
        _style(axes)
        return figure

    if baseline_centre is not None and np.isfinite(baseline_centre):
        axes.axhline(baseline_centre, color="#666666", lw=0.9, ls="--",
                     label="personal baseline")
        if baseline_spread and np.isfinite(baseline_spread) and baseline_spread > 0:
            axes.axhspan(baseline_centre - baseline_spread,
                         baseline_centre + baseline_spread,
                         color="#666666", alpha=0.10, lw=0)

    axes.plot(frame["session_date"], frame[metric], "-", color="#3182bd", lw=1.3,
              zorder=2)

    confident = frame[frame.get(low_confidence_column, 0) == 0]
    provisional = frame[frame.get(low_confidence_column, 0) == 1]
    axes.scatter(confident["session_date"], confident[metric], s=42,
                 color="#3182bd", zorder=3, label="session")
    if not provisional.empty:
        axes.scatter(provisional["session_date"], provisional[metric], s=42,
                     facecolors="none", edgecolors="#3182bd", linewidths=1.4,
                     zorder=3, label="low confidence")

    if flagged_dates is not None and len(flagged_dates):
        flagged = frame[frame["session_date"].isin(list(flagged_dates))]
        if not flagged.empty:
            axes.scatter(flagged["session_date"], flagged[metric], s=150,
                         marker="o", facecolors="none", edgecolors="#d7301f",
                         linewidths=2.0, zorder=4, label="flagged")

    axes.set_ylabel(label, fontsize=9)
    axes.legend(loc="best", fontsize=8, framealpha=0.9)
    figure.autofmt_xdate(rotation=30)
    _style(axes)
    return figure
