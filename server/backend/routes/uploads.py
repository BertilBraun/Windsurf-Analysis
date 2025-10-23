from __future__ import annotations

import hashlib
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
import modal
from pydantic import BaseModel, Field

from server.backend.auth import authenticate_user
from server.backend.database.accessor import DatabaseAccessor, timestamp_now
from server.backend.database.db import get_db
from server.backend.models import JobStatus, User
from server.inference.src.util.timing import timeit


router = APIRouter(prefix='/jobs', tags=['uploads'])


class UploadInitResponse(BaseModel):
    resume_from_part: int = Field(ge=0)


class UploadMeta(BaseModel):
    total_size: int = Field(ge=0)
    chunk_size: int = Field(gt=0)
    total_parts: int = Field(gt=0)
    file_name: str
    mime_type: str = 'video/mp4'
    yolo_model: str


def _upload_dir_for_job(job_id: str) -> Path:
    base = Path('/data/uploads') / job_id
    (base / 'parts').mkdir(parents=True, exist_ok=True)
    return base


@router.post('/{job_id}/upload/init', response_model=UploadInitResponse)
async def upload_init(
    job_id: str,
    total_size: int = Form(...),
    chunk_size: int = Form(...),
    total_parts: int = Form(...),
    file_name: str = Form(...),
    mime_type: str = Form('video/mp4'),
    yolo_model: str = Form(...),
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    job = await db.get_job_by_id_and_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    if job.status not in (JobStatus.pending,):
        raise HTTPException(status_code=409, detail='Job not in a state that accepts uploads')

    meta = UploadMeta(
        total_size=int(total_size),
        chunk_size=int(chunk_size),
        total_parts=int(total_parts),
        file_name=file_name,
        mime_type=mime_type,
        yolo_model=yolo_model,
    )

    upload_dir = _upload_dir_for_job(job_id)
    meta_path = upload_dir / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(meta.model_dump_json())

    parts_dir = upload_dir / 'parts'
    existing_parts = {int(p.stem.split('_')[-1]) for p in parts_dir.glob('part_*.bin') if p.is_file()}
    resume_from = 0
    while resume_from in existing_parts:
        resume_from += 1

    from server.main_backend_frontend import volume

    volume.commit()

    return UploadInitResponse(resume_from_part=resume_from)


@router.post('/{job_id}/upload/part')
async def upload_part(
    job_id: str,
    part_index: int = Form(...),
    chunk: UploadFile = File(...),
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    job = await db.get_job_by_id_and_user(job_id, user)
    if job is None:
        raise HTTPException(status_code=404, detail='Not found')

    if job.status not in (JobStatus.pending,):
        raise HTTPException(status_code=409, detail='Job not in a state that accepts uploads')

    if int(part_index) < 0:
        raise HTTPException(status_code=400, detail='Invalid part index')

    upload_dir = _upload_dir_for_job(job_id)
    parts_dir = upload_dir / 'parts'
    target_path = parts_dir / f'part_{int(part_index)}.bin'

    content = await chunk.read()
    tmp_path = target_path.with_suffix('.tmp')
    with open(tmp_path, 'wb') as f:
        f.write(content)
    os.replace(tmp_path, target_path)

    from server.main_backend_frontend import volume

    volume.commit()

    return {'ok': True}


def _cleanup_upload_parts_dir(parts_dir: Path):
    try:
        for p in parts_dir.glob('*'):
            p.unlink(missing_ok=True)
        parts_dir.rmdir()
    except Exception:
        pass


def _cleanup_upload_dir(upload_dir: Path):
    _cleanup_upload_parts_dir(upload_dir / 'parts')
    (upload_dir / 'meta.json').unlink(missing_ok=True)
    upload_dir.rmdir()


@router.post('/{job_id}/upload/complete')
async def upload_complete(
    job_id: str,
    db: DatabaseAccessor = Depends(get_db),
    user: User = Depends(authenticate_user),
):
    existing_job = await db.get_job_by_id_and_user(job_id, user)
    if existing_job is None:
        raise HTTPException(status_code=404, detail='Not found')

    job = existing_job
    if job.status not in (JobStatus.pending,):
        raise HTTPException(status_code=409, detail='Job not in a state that accepts uploads')

    upload_dir = _upload_dir_for_job(job_id)
    meta_path = upload_dir / 'meta.json'
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail='Upload not initialized')

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = UploadMeta.model_validate_json(f.read())

    parts_dir = upload_dir / 'parts'
    part_paths = [parts_dir / f'part_{i}.bin' for i in range(int(meta.total_parts))]
    missing = [str(p.name) for p in part_paths if not p.exists()]
    if missing:
        _cleanup_upload_dir(upload_dir)
        raise HTTPException(status_code=400, detail=f'Missing parts: {", ".join(missing)}')

    final_path = Path(f'/data/{job_id}.mp4')
    with timeit('concat_parts'):
        with open(final_path, 'wb') as out:
            for p in part_paths:
                with open(p, 'rb') as inp:
                    while True:
                        block = inp.read(1024 * 1024)
                        if not block:
                            break
                        out.write(block)

    _cleanup_upload_dir(upload_dir)

    from server.main_backend_frontend import volume

    volume.commit()

    # Compute checksum via streaming
    hasher = hashlib.sha256()
    with open(final_path, 'rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    ac_checksum = hasher.hexdigest()

    job.ac_checksum_sha256 = ac_checksum
    job.size_bytes = final_path.stat().st_size
    job.mime_type = meta.mime_type or 'video/mp4'
    job.ac_storage_url = 'N/A'

    job.status = JobStatus.orientation
    job.started_at = timestamp_now()
    await db.flush()

    with timeit('spawn_stabilization'):
        StabilizationModel = modal.Cls.from_name('windsurf-analysis', 'StabilizationModel')
        StabilizationModel().stabilize_and_enqueue.spawn(job_id=str(job.id), yolo_model=meta.yolo_model)

    from server.main_backend_frontend import volume

    volume.commit()

    return {'ok': True}
