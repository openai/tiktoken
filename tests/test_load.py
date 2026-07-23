from tiktoken.load import read_file


class FakeResponse:
    content = b"fake content"

    def raise_for_status(self):
        pass


class FakeSession:
    def __init__(self):
        self.requested_urls = []

    def get(self, url):
        self.requested_urls.append(url)
        return FakeResponse()


def test_read_file_uses_provided_session():
    session = FakeSession()
    url = "https://example.com/some-file"

    data = read_file(url, session=session)

    assert data == b"fake content"
    assert session.requested_urls == [url]
