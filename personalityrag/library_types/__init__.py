"""Database type implementations and registration contracts."""

from ..database_types import database_type_registry
from .livingmemory_v8 import driver as livingmemory_v8_driver
from .text_media_v1 import driver as text_media_v1_driver


database_type_registry.register(livingmemory_v8_driver)
database_type_registry.register(text_media_v1_driver)
