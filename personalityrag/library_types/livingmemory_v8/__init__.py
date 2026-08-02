"""LivingMemory database v8 implementation."""

from .driver import driver
from . import task_types as _task_types  # noqa: F401


DATABASE_TYPE = driver.descriptor.id
DATABASE_CATEGORY = driver.descriptor.category
