from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


class UploadMeta(BaseModel):
    total_size: int
    chunk_size: int
    total_parts: int
    file_name: str
    mime_type: str = 'video/mp4'
    yolo_model: str


def upload_dir_for_job(job_id: str) -> Path:
    base = Path('/data/uploads') / job_id
    (base / 'parts').mkdir(parents=True, exist_ok=True)
    return base


def write_meta(job_id: str, meta: UploadMeta) -> None:
    upload_dir = upload_dir_for_job(job_id)
    meta_path = upload_dir / 'meta.json'
    meta_path.write_text(meta.model_dump_json(), encoding='utf-8')


def read_meta(job_id: str) -> UploadMeta:
    upload_dir = upload_dir_for_job(job_id)
    meta_path = upload_dir / 'meta.json'
    if not meta_path.exists():
        raise ValueError('Upload not initialized')
    return UploadMeta.model_validate_json(meta_path.read_text(encoding='utf-8'))


def compute_resume_from_part(job_id: str) -> int:
    parts_dir = upload_dir_for_job(job_id) / 'parts'
    existing_parts = {int(p.stem.split('_')[-1]) for p in parts_dir.glob('part_*.bin') if p.is_file()}
    resume_from = 0
    while resume_from in existing_parts:
        resume_from += 1
    return resume_from


def write_part(job_id: str, part_index: int, content: bytes) -> None:
    if part_index < 0:
        raise ValueError('Invalid part index')
    upload_dir = upload_dir_for_job(job_id)
    parts_dir = upload_dir / 'parts'
    target_path = parts_dir / f'part_{int(part_index)}.bin'
    target_path.write_bytes(content)


def _cleanup_upload_parts_dir(parts_dir: Path) -> None:
    try:
        for p in parts_dir.glob('*'):
            p.unlink(missing_ok=True)
        parts_dir.rmdir()
    except Exception:
        pass


def cleanup_upload_dir(job_id: str) -> None:
    upload_dir = upload_dir_for_job(job_id)
    _cleanup_upload_parts_dir(upload_dir / 'parts')
    try:
        (upload_dir / 'meta.json').unlink(missing_ok=True)
    except Exception:
        pass
    try:
        upload_dir.rmdir()
    except Exception:
        pass


def concat_parts_to_final(job_id: str, meta: UploadMeta) -> Path:
    upload_dir = upload_dir_for_job(job_id)
    parts_dir = upload_dir / 'parts'
    part_paths = [parts_dir / f'part_{i}.bin' for i in range(int(meta.total_parts))]
    missing = [p.name for p in part_paths if not p.exists()]
    if missing:
        cleanup_upload_dir(job_id)
        raise ValueError(f'Missing parts: {", ".join(missing)}')

    final_path = Path(f'/data/{job_id}.mp4')
    with final_path.open('wb') as out:
        for p in part_paths:
            with p.open('rb') as inp:
                while True:
                    block = inp.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)

    cleanup_upload_dir(job_id)
    return final_path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()
