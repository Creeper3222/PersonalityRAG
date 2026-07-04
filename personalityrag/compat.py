"""Compatibility markers for external memory formats.

PersonalityRAG tracks LivingMemory by its database schema version, not by the
AstrBot plugin release number. Plugin releases may change UI or adapter logic
without changing the persisted memory format; the schema version is the stable
boundary that matters for import, migration and retrieval compatibility.
"""

LIVINGMEMORY_DATABASE_VERSION = 8
LIVINGMEMORY_DATABASE_VERSION_LABEL = f"LivingMemory DB v{LIVINGMEMORY_DATABASE_VERSION}"


def default_library_metadata() -> dict[str, int]:
    """Metadata stamped onto each library record at creation/migration time."""

    return {"livingmemory_database_version": LIVINGMEMORY_DATABASE_VERSION}
