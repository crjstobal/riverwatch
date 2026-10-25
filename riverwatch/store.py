"""Persistence for agent state, traces and status files: local folder or S3, same interface."""
from __future__ import annotations

import json
from pathlib import Path


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def get_json(self, key: str, default=None):
        p = self.root / key
        return json.loads(p.read_text()) if p.exists() else default

    def put_json(self, key: str, obj, cache_seconds: int = 0):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, default=str))

    def put_bytes(self, key: str, data: bytes, content_type: str, cache_seconds: int = 0):
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


class S3Store:
    def __init__(self, bucket: str, prefix: str = "", session=None):
        import boto3
        self.s3 = (session or boto3.Session()).client("s3")
        self.bucket, self.prefix = bucket, prefix

    def get_json(self, key: str, default=None):
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return default
        return json.loads(body)

    def put_json(self, key: str, obj, cache_seconds: int = 0):
        self.put_bytes(key, json.dumps(obj, default=str).encode(), "application/json", cache_seconds)

    def put_bytes(self, key: str, data: bytes, content_type: str, cache_seconds: int = 0):
        self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data, ContentType=content_type,
                           CacheControl=f"max-age={cache_seconds}" if cache_seconds else "no-cache")
