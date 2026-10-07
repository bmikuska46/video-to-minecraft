from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import Settings


@dataclass(frozen=True)
class SignedUrl:
    url: str
    expires_at: datetime
    required_headers: dict[str, str]


@dataclass(frozen=True)
class StoredObject:
    byte_size: int
    sha256: str | None


class ObjectNotFoundError(FileNotFoundError):
    pass


class ObjectStorage(Protocol):
    def ready(self) -> None: ...

    def sign_put(
        self, key: str, *, content_type: str, byte_size: int, sha256: str | None
    ) -> SignedUrl: ...

    def sign_get(self, key: str) -> SignedUrl: ...

    def head(self, key: str) -> StoredObject: ...

    def read_bytes(self, key: str) -> bytes: ...

    def download_file(self, key: str, destination: Path) -> None: ...

    def upload_file(
        self, key: str, source: Path, *, content_type: str, sha256: str
    ) -> None: ...


class S3ObjectStorage:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.ttl = settings.signed_url_ttl_seconds
        client_options = {
            "region_name": settings.s3_region,
            "aws_access_key_id": settings.s3_access_key_id,
            "aws_secret_access_key": settings.s3_secret_access_key,
            "config": Config(signature_version="s3v4"),
        }
        self.client = boto3.client("s3", endpoint_url=settings.s3_endpoint_url, **client_options)
        public_endpoint = settings.s3_public_endpoint_url or settings.s3_endpoint_url
        self.signing_client = (
            self.client
            if public_endpoint == settings.s3_endpoint_url
            else boto3.client("s3", endpoint_url=public_endpoint, **client_options)
        )

    def ready(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)

    def sign_put(
        self, key: str, *, content_type: str, byte_size: int, sha256: str | None
    ) -> SignedUrl:
        metadata = {"sha256": sha256} if sha256 else {}
        params = {
            "Bucket": self.bucket,
            "Key": key,
            "ContentType": content_type,
            "ContentLength": byte_size,
            "Metadata": metadata,
            "IfNoneMatch": "*",
        }
        url = self.signing_client.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=self.ttl, HttpMethod="PUT"
        )
        headers = {
            "content-type": content_type,
            "content-length": str(byte_size),
            "if-none-match": "*",
        }
        headers.update({f"x-amz-meta-{name}": value for name, value in metadata.items()})
        return SignedUrl(url, datetime.now(timezone.utc) + timedelta(seconds=self.ttl), headers)

    def sign_get(self, key: str) -> SignedUrl:
        url = self.signing_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=self.ttl,
            HttpMethod="GET",
        )
        return SignedUrl(url, datetime.now(timezone.utc) + timedelta(seconds=self.ttl), {})

    def head(self, key: str) -> StoredObject:
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise ObjectNotFoundError(key) from error
            raise
        return StoredObject(
            byte_size=int(response["ContentLength"]),
            sha256=response.get("Metadata", {}).get("sha256"),
        )

    def read_bytes(self, key: str) -> bytes:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise ObjectNotFoundError(key) from error
            raise
        return response["Body"].read()

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, key, str(destination))
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                raise ObjectNotFoundError(key) from error
            raise

    def upload_file(
        self, key: str, source: Path, *, content_type: str, sha256: str
    ) -> None:
        self.client.upload_file(
            str(source),
            self.bucket,
            key,
            ExtraArgs={"ContentType": content_type, "Metadata": {"sha256": sha256}},
        )
