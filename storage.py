"""S3-compatible (RustFS) storage client for datasets and artifacts."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class StorageClient:
    """Thin wrapper around boto3 for dataset fetch + artifact upload."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        dataset_bucket: str,
        model_bucket: str,
    ) -> None:
        try:
            import boto3
            from botocore.client import Config as BotoConfig
        except ImportError as e:
            raise RuntimeError(
                "boto3 is required — `uv pip install boto3`"
            ) from e

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )
        self.dataset_bucket = dataset_bucket
        self.model_bucket = model_bucket
        self.endpoint = endpoint
        # Kept for building per-thread clients in parallel downloads.
        self._access_key = access_key
        self._secret_key = secret_key

    # ---------------------------------------------------------------- datasets
    def download_dataset(self, dataset_id: str, dest_dir: Path) -> Path:
        """Download all dataset files under `datasets/{dataset_id}/` to dest_dir.

        Downloads run in parallel. A LeRobot dataset is hundreds of small
        objects (one video + one parquet per episode) — a G1 PickBottle run
        is ~685 files / 3.5 GB. Boto's per-file `download_file` overhead (HEAD
        + ranged GETs) dominates, so a sequential loop crawls at ~4 s/file
        (hours for the full set). The S3 store serves requests concurrently,
        so a thread pool cuts that ~6-10x. Each thread gets its own boto3
        client (clients are not thread-safe to share for transfers).

        Returns the destination directory.
        """
        from concurrent.futures import ThreadPoolExecutor
        import threading

        dest_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{dataset_id}/"
        log.info("Downloading dataset %s from %s/%s", dataset_id, self.dataset_bucket, prefix)

        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.dataset_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key[len(prefix):]:  # skip the folder placeholder
                    keys.append(key)

        if not keys:
            raise FileNotFoundError(
                f"No objects found under {self.dataset_bucket}/{prefix}"
            )

        # One client per worker thread (boto3 clients aren't safe to share
        # across concurrent transfers). Built lazily and reused per thread.
        tl = threading.local()

        def _client_for_thread():
            c = getattr(tl, "client", None)
            if c is None:
                import boto3
                from botocore.client import Config as BotoConfig
                c = boto3.client(
                    "s3",
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self._access_key,
                    aws_secret_access_key=self._secret_key,
                    config=BotoConfig(signature_version="s3v4"),
                    region_name="us-east-1",
                )
                tl.client = c
            return c

        def _download_one(key: str) -> None:
            rel = key[len(prefix):]  # mirror the internal layout
            local = dest_dir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            _client_for_thread().download_file(self.dataset_bucket, key, str(local))

        with ThreadPoolExecutor(max_workers=16) as pool:
            # list(...) re-raises the first download failure
            list(pool.map(_download_one, keys))

        log.info("Downloaded %d file(s) to %s", len(keys), dest_dir)
        return dest_dir

    # ------------------------------------------------------------- model artifacts
    def upload_artifact(self, job_id: str, local_path: Path, name: str) -> str:
        """Upload a file to models/{job_id}/{name} — returns an s3:// URI.

        Multi-GB checkpoints are uploaded with a large (512 MB) multipart
        chunk size. boto3's 8 MB default turns a ~7 GB artifact into ~900
        parts, and moto's in-memory CompleteMultipartUpload 500s while
        stitching that many parts back together (observed on the 17 GB
        full-finetune tar). Fewer, larger parts keep the Complete call
        well within moto's limits.
        """
        from boto3.s3.transfer import TransferConfig

        key = f"{job_id}/{name}"
        size = local_path.stat().st_size
        log.info(
            "Uploading artifact %s → %s/%s (%d bytes)",
            local_path.name,
            self.model_bucket,
            key,
            size,
        )
        cfg = TransferConfig(
            multipart_threshold=512 * 1024 * 1024,
            multipart_chunksize=512 * 1024 * 1024,
            max_concurrency=4,
            use_threads=True,
        )
        self._client.upload_file(str(local_path), self.model_bucket, key, Config=cfg)
        return f"s3://{self.model_bucket}/{key}"

    # ---------------------------------------------------------------- ensure
    def ensure_model_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self.model_bucket)
        except Exception:
            log.info("Creating model bucket: %s", self.model_bucket)
            self._client.create_bucket(Bucket=self.model_bucket)
