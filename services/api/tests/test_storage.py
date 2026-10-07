from unittest.mock import Mock, patch

from app.config import Settings
from app.storage import S3ObjectStorage


def test_public_endpoint_is_used_only_for_signed_client_urls() -> None:
    internal = Mock()
    public = Mock()
    public.generate_presigned_url.return_value = "http://192.168.1.20:9000/signed"

    with patch("app.storage.boto3.client", side_effect=[internal, public]) as make_client:
        storage = S3ObjectStorage(
            Settings(
                s3_endpoint_url="http://minio:9000",
                s3_public_endpoint_url="http://192.168.1.20:9000",
            )
        )
        signed = storage.sign_get("scans/example/preview.glb")

    assert signed.url == "http://192.168.1.20:9000/signed"
    assert make_client.call_count == 2
    assert make_client.call_args_list[0].kwargs["endpoint_url"] == "http://minio:9000"
    assert make_client.call_args_list[1].kwargs["endpoint_url"] == "http://192.168.1.20:9000"
    public.generate_presigned_url.assert_called_once()
    internal.generate_presigned_url.assert_not_called()
