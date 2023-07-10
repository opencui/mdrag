import io
import zipfile
import tarfile
import pytest

from main import init_app


@pytest.fixture
def app(loop, aiohttp_client):
    return loop.run_until_complete(aiohttp_client(init_app()))


@pytest.mark.asyncio
async def test_hello_world(app):
    resp = await app.get("/")
    assert resp.status == 200
    resp = await resp.text()
    assert resp == "Hello, world"


zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
    zip_file.writestr("main.py", open("main.py").read())

tar_buffer = io.BytesIO()
with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
    tar.add("main.py")


@pytest.mark.asyncio
async def test_upload_file(app):
    test_list = [
        {},
        {
            "fileType": "text",
            "fileData": "hello world"
        },
        {
            "fileType": "tar",
            "fileData": tar_buffer.getvalue()
        },
        {
            "fileType": "zip",
            "fileData": zip_buffer.getvalue()
        },
        {
            "fileType":
            "url+text",
            "fileData":
            "https://raw.githubusercontent.com/framely/actions-feishu-bot-hook/main/action.yaml"
        },
        {
            "fileType":
            "url+tar",
            "fileData":
            "https://github.com/framely/actions-init-tools/archive/refs/tags/test-download.tar.gz",
        },
        {
            "fileType":
            "url+zip",
            "fileData":
            "https://github.com/framely/actions-feishu-bot-hook/archive/refs/heads/main.zip",
        },
    ]
    for i, d in enumerate(test_list):
        resp = await app.post("/upload-file", data=d)
        assert resp.status == 200
        resp = await resp.json()
        if i == 0:
            assert resp.get("errMsg") == "fileType type is not str"
            continue
        else:
            assert resp.get("errMsg") == None


@pytest.mark.asyncio
async def test_query(app):
    uid = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    data = {"uid": uid, "input": "中国有多大"}
    resp = await app.post("/query", data=data)
    assert resp.status == 200
    resp = await resp.json()
    assert resp.get("result") == "\n中国面积约960万平方公里。"
