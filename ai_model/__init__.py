"""Machine-learning components for the real-time intrusion detection system.

The package deliberately keeps the training representation (NSL-KDD) and the
live-packet representation separate.  ``schema`` documents the bridge between
the two, while ``inference`` is the only module an application needs to load a
deployed model artifact.
"""

from .schema import NSL_KDD_FEATURES

__all__ = ["NSL_KDD_FEATURES"]
