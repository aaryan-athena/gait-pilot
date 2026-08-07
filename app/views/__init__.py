"""One module per page of the app.

Each exposes ``render(cfg)``. The uniform signature is what lets
:mod:`app.main` treat navigation as a lookup table rather than a chain of
conditionals.
"""

from . import analyse, calibration, limitations, trends

__all__ = ["analyse", "calibration", "limitations", "trends"]
