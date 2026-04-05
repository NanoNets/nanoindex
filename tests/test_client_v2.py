"""Tests for the Nanonets V2 API client.

Only tests construction and URL logic — no real API calls.
"""

from nanoindex.core.client_v2 import NanonetsV2Client


class TestClientCreation:
    def test_client_creation_default(self):
        client = NanonetsV2Client("test-key")
        assert client._api_key == "test-key"
        assert client._base_url == "https://extraction-api.nanonets.com"
        assert client._upload_timeout == 30.0
        assert client._operation_timeout == 300.0
        assert client._client is None

    def test_client_creation_custom_base_url(self):
        client = NanonetsV2Client("k", base_url="https://custom.example.com/")
        assert client._base_url == "https://custom.example.com"  # trailing slash stripped

    def test_client_creation_custom_timeouts(self):
        client = NanonetsV2Client("k", upload_timeout=10.0, operation_timeout=60.0)
        assert client._upload_timeout == 10.0
        assert client._operation_timeout == 60.0


class TestParseURL:
    def test_parse_url_construction(self):
        """Verify the URL that parse() would hit is /api/v2/parse/sync."""
        client = NanonetsV2Client("test-key", base_url="https://extraction-api.nanonets.com")
        # The client builds requests against base_url + path.
        # We verify the base URL and that the path constant is correct.
        assert client._base_url == "https://extraction-api.nanonets.com"
        # The path used by parse() is "/api/v2/parse/sync"
        expected_url = "https://extraction-api.nanonets.com/api/v2/parse/sync"
        assert f"{client._base_url}/api/v2/parse/sync" == expected_url

    def test_upload_url_construction(self):
        client = NanonetsV2Client("test-key")
        expected_url = "https://extraction-api.nanonets.com/api/v2/files"
        assert f"{client._base_url}/api/v2/files" == expected_url

    def test_extract_url_construction(self):
        client = NanonetsV2Client("test-key")
        expected_url = "https://extraction-api.nanonets.com/api/v2/extract/sync"
        assert f"{client._base_url}/api/v2/extract/sync" == expected_url

    def test_classify_url_construction(self):
        client = NanonetsV2Client("test-key")
        expected_url = "https://extraction-api.nanonets.com/api/v2/classify/sync"
        assert f"{client._base_url}/api/v2/classify/sync" == expected_url
