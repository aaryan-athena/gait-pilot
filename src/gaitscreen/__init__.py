"""gaitscreen -- video-based gait screening and trend monitoring.

This is a **screening and trend-monitoring tool, not a diagnostic instrument**.
It is built to flag concerning *changes* in an individual's walking over time,
and it prioritises consistency and repeatability of measurement over absolute
accuracy. Any absolute value it reports should be read as indicative.

Every clinical threshold in ``config/default.yaml`` is illustrative and must be
reviewed against current geriatric literature before real-world use.
"""

from .config import Config
from .version import ALGO_VERSION

__all__ = ["Config", "ALGO_VERSION"]
