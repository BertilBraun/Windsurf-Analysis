from __future__ import annotations
import json
import os

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile

import modal

from config import settings

from .clients.cloudrun_client import CloudRunClient, CloudRunError
from .services.upload_service import (
    UploadMeta,
    compute_resume_from_part,
    concat_parts_to_final,
    read_meta,
    sha256_file,
    write_meta,
    write_part,
)


router = APIRouter(prefix='/jobs', tags=['uploads'])
cr = CloudRunClient()


def _require_bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Missing Authorization: Bearer <Firebase ID token>')
    return authorization


def _reload_volume() -> None:
    from video_processing.main_api import volume

    volume.reload()


def _commit_volume() -> None:
    from video_processing.main_api import volume

    volume.commit()


def _verify_job_authorization(job_id: str, authorization: str | None) -> None:
    try:
        cr.verify_job(job_id, _require_bearer(authorization), required_statuses=['pending'])
    except CloudRunError as e:
        raise HTTPException(status_code=e.status_code, detail=e.body)


@router.post('/{job_id}/upload/init')
async def upload_init(
    job_id: str,
    total_size: int = Form(...),
    chunk_size: int = Form(...),
    total_parts: int = Form(...),
    file_name: str = Form(...),
    mime_type: str = Form('video/mp4'),
    yolo_model: str = Form(...),
    authorization: str | None = Header(default=None),
):
    _verify_job_authorization(job_id, authorization)

    meta = UploadMeta(
        total_size=int(total_size),
        chunk_size=int(chunk_size),
        total_parts=int(total_parts),
        file_name=file_name,
        mime_type=mime_type,
        yolo_model=yolo_model,
    )
    write_meta(job_id, meta)
    _commit_volume()

    resume_from = compute_resume_from_part(job_id)
    return {'resume_from_part': resume_from}


@router.post('/{job_id}/upload/part')
async def upload_part(
    job_id: str,
    part_index: int = Form(...),
    chunk: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    _verify_job_authorization(job_id, authorization)

    if part_index < 0:
        raise HTTPException(status_code=400, detail='Part index must be non-negative')

    content = await chunk.read()
    write_part(job_id, part_index, content)

    _commit_volume()
    return {'ok': True}


@router.post('/{job_id}/upload/complete')
async def upload_complete(job_id: str, authorization: str | None = Header(default=None)):
    _verify_job_authorization(job_id, authorization)

    _reload_volume()

    try:
        meta = read_meta(job_id)
        final_path = concat_parts_to_final(job_id, meta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    _commit_volume()

    ac_checksum = sha256_file(final_path)

    # Persist canonical metadata in Cloud Run / Firestore
    try:
        cr.mark_uploaded(
            job_id,
            ac_checksum_sha256=ac_checksum,
            size_bytes=int(final_path.stat().st_size),
            mime_type=meta.mime_type or 'video/mp4',
            ac_storage_url=str(final_path),
            authorization=_require_bearer(authorization),
        )
    except CloudRunError as e:
        raise HTTPException(status_code=e.status_code, detail=e.body)

    # Kick off existing Modal pipeline (keeps compute on Modal)
    StabilizationModel = modal.Cls.from_name('windsurf-analysis', 'StabilizationModel')
    StabilizationModel().stabilize_and_enqueue.spawn(job_id=str(job_id), yolo_model=meta.yolo_model)

    return {'ok': True}


@router.post('/{job_id}/complete')
async def complete_job(
    job_id: str,
    status: str = Form(...),
    results_json: str | None = Form(None),
    error_message: str | None = Form(None),
    modal_shared_secret: str = Header(default=None),
):
    """Called when processing is finished."""
    # verify, that the modal shared secret is present
    if not modal_shared_secret or modal_shared_secret != settings.modal_shared_secret:
        raise HTTPException(status_code=401, detail='Invalid modal shared secret')
    _reload_volume()

    try:
        os.remove(f'/data/{job_id}_upright.mp4')
        _commit_volume()
    except Exception as e:
        print(f'Error removing video: {e}')

    results = None
    if results_json:
        try:
            results = json.loads(results_json)
        except Exception:
            raise HTTPException(status_code=400, detail='Invalid results_json')

    try:
        cr.set_results(
            job_id,
            status=status,
            results=results,
            error_message=error_message,
        )
    except CloudRunError as e:
        raise HTTPException(status_code=e.status_code, detail=e.body)

    return {'ok': True}
