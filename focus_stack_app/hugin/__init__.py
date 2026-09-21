"""Hugin/Enfuse command discovery and subprocess adapters."""

from .hugin_locator import HuginInstallation, HuginLocator, HuginToolNotFound, find_hugin, locate_hugin
from .align import AlignConfig, AlignmentError, AlignmentResult, AlignImageStack, AlignImageStackRunner, align_images
from .enfuse import EnfuseConfig, EnfuseError, EnfuseResult, Enfuser, EnfuseRunner, enfuse_images
from .output_encoder import (
    OutputConfig,
    OutputCollisionError,
    OutputEncodingError,
    OutputFormat,
    output_path_for,
    encode_output,
)

__all__ = [
    "AlignConfig",
    "AlignImageStack",
    "AlignImageStackRunner",
    "AlignmentError",
    "AlignmentResult",
    "EnfuseConfig",
    "EnfuseError",
    "EnfuseResult",
    "Enfuser",
    "EnfuseRunner",
    "HuginInstallation",
    "HuginLocator",
    "HuginToolNotFound",
    "find_hugin",
    "locate_hugin",
    "OutputCollisionError",
    "OutputConfig",
    "OutputEncodingError",
    "OutputFormat",
    "encode_output",
    "output_path_for",
    "align_images",
    "enfuse_images",
]
