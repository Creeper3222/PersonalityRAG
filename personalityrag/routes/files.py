from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from ..application_context import current_context
from ..http_shared import require_admin_auth


router = APIRouter()


class FilePathRequest(BaseModel):
    root: str = ""
    path: str = ""
    password: str = ""


class FileWriteRequest(FilePathRequest):
    content: str


class FileCreateRequest(FilePathRequest):
    type: Literal["file", "directory"] = "file"


class FilePathsRequest(BaseModel):
    root: str = ""
    paths: list[str] = Field(default_factory=list)
    password: str = ""


class FileMoveRequest(FilePathsRequest):
    target_path: str = ""


class FileRenameRequest(FilePathRequest):
    name: str


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _download_response(path: Path, name: str, temporary: bool) -> FileResponse:
    background = None
    if temporary:
        background = BackgroundTask(
            current_context().file_manager.cleanup_archive,
            path,
        )
    return FileResponse(
        path,
        filename=name,
        media_type="application/zip" if temporary else "application/octet-stream",
        background=background,
    )


@router.get("/api/v1/files", dependencies=[Depends(require_admin_auth)])
async def list_files(
    root: str = Query(default=""),
    path: str = Query(default=""),
):
    return await current_context().file_manager.list_directory(root, path)


@router.post("/api/v1/files/read", dependencies=[Depends(require_admin_auth)])
async def read_file(payload: FilePathRequest, request: Request):
    return await current_context().file_manager.read_text(
        payload.root,
        payload.path,
        client_ip=_client_ip(request),
        password=payload.password,
    )


@router.post("/api/v1/files/write", dependencies=[Depends(require_admin_auth)])
async def write_file(payload: FileWriteRequest, request: Request):
    return await current_context().file_manager.write_text(
        payload.root,
        payload.path,
        payload.content,
        client_ip=_client_ip(request),
        password=payload.password,
    )


@router.post("/api/v1/files/create", dependencies=[Depends(require_admin_auth)])
async def create_file_item(payload: FileCreateRequest):
    return await current_context().file_manager.create_item(
        payload.root,
        payload.path,
        payload.type,
    )


@router.post("/api/v1/files/upload", dependencies=[Depends(require_admin_auth)])
async def upload_files(
    root: str = Query(default=""),
    path: str = Query(default=""),
    files: list[UploadFile] = File(default=[]),
):
    return await current_context().file_manager.upload_files(root, path, files)


@router.post("/api/v1/files/delete", dependencies=[Depends(require_admin_auth)])
async def delete_file_items(payload: FilePathsRequest):
    return await current_context().file_manager.delete_items(
        payload.root,
        payload.paths,
    )


@router.post("/api/v1/files/move", dependencies=[Depends(require_admin_auth)])
async def move_file_items(payload: FileMoveRequest):
    return await current_context().file_manager.move_items(
        payload.root,
        payload.paths,
        payload.target_path,
    )


@router.post("/api/v1/files/rename", dependencies=[Depends(require_admin_auth)])
async def rename_file_item(payload: FileRenameRequest):
    return await current_context().file_manager.rename_item(
        payload.root,
        payload.path,
        payload.name,
    )


@router.get("/api/v1/files/preview", dependencies=[Depends(require_admin_auth)])
async def preview_file(
    root: str = Query(default=""),
    path: str = Query(default=""),
):
    target, media_type = current_context().file_manager.preview_file(root, path)
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "no-store"})


@router.get("/api/v1/files/download", dependencies=[Depends(require_admin_auth)])
async def download_file(
    request: Request,
    root: str = Query(default=""),
    path: str = Query(default=""),
):
    target, name, temporary = await current_context().file_manager.prepare_download(
        root,
        [path],
        client_ip=_client_ip(request),
        password="",
    )
    return _download_response(target, name, temporary)


@router.post("/api/v1/files/download", dependencies=[Depends(require_admin_auth)])
async def download_files(payload: FilePathsRequest, request: Request):
    target, name, temporary = await current_context().file_manager.prepare_download(
        payload.root,
        payload.paths,
        client_ip=_client_ip(request),
        password=payload.password,
    )
    return _download_response(target, name, temporary)
