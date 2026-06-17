"""Utilities for extracting visible surface elements from a view direction."""

from .visibility import Face, VisibilityResult, extract_visible_faces, is_front_facing

__all__ = [
    "Face",
    "VisibilityResult",
    "extract_visible_faces",
    "is_front_facing",
]
