from __future__ import annotations


PRODUCT_NAME = "PersonalityRAG"
PLATFORM_NAME = "linux-docker"
RELEASE_ROOT_NAME = "PersonalityRAG-linux"
RELEASE_ASSET_PREFIX = "PersonalityRAG-linux"
DOCKER_REPOSITORY = "138763327/personalityrag-linux"
VERSION = "0.1.2"
TAG_NAME = f"v{VERSION}"


def display_version() -> str:
    return TAG_NAME


def release_asset_name(tag_name: str = TAG_NAME) -> str:
    return f"{RELEASE_ASSET_PREFIX}-{tag_name}.zip"
