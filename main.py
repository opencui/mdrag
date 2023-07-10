#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import logging

sys.path.append(".")

from aiohttp import web, web_request
import task
import utils

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

routes = web.RouteTableDef()


@routes.get("/")
async def hello(_: web.Request):
    return web.Response(text="Hello, world")


# post form {fileType: text|tar|zip|url+text|url+tar|url+zip, fileData: str|bytes|bytes|url|url|url}
# curl -v -d 'fileType=text' -d 'fileData=hello world' http://127.0.0.1:8080/upload-file
# curl -v -d 'fileType=url%2Btext' -d 'fileData=https://www.baidu.com/' http://127.0.0.1:8080/upload-file
# curl -v -d 'fileType=url%2Bzip'  -d 'fileData=https://github.com/framely/actions-feishu-bot-hook/archive/refs/heads/main.zip' http://127.0.0.1:8080/upload-file
# curl -v -d 'fileType=url%2Btar'  -d 'fileData=https://github.com/framely/actions-init-tools/archive/refs/tags/test-download.tar.gz' http://127.0.0.1:8080/upload-file
@routes.post("/upload-file")
async def upload_file(request: web.Request):
    req = await request.post()
    fileType = req.get("fileType")
    fileData = req.get("fileData")

    if type(fileType) != str:
        return web.json_response({"errMsg": f'fileType type is not str'})

    if type(fileData) not in [str, web_request.FileField]:
        return web.json_response(
            {"errMsg": f'fileData type is not str | bytes'})

    if type(fileData) is web_request.FileField:
        fileData = fileData.file

    try:
        uid = utils.sha256(fileData)
        await task.task_build_index(fileType, uid, fileData)
    except Exception as e:
        return web.json_response({"errMsg": str(e)})

    resp = {"uid": uid}
    return web.json_response(resp)


# post form {uid: str, input: str}
# curl -v -d 'uid=b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9' -d 'input=中国有多大' http://127.0.0.1:8080/query
@routes.post("/query")
async def query(request: web.Request):
    req = await request.post()
    uid = req.get("uid")
    input = req.get("input")

    if type(uid) != str:
        return web.json_response({"errMsg": f'uid type is not str'})

    if type(input) != str:
        return web.json_response({"errMsg": f'input type is not str'})

    try:
        result = await task.run_on_executor(task.task_query, uid, input)
    except Exception as e:
        return web.json_response({"errMsg": str(e)})

    resp = {"result": str(result)}
    return web.json_response(resp)


def init_app():
    app = web.Application()
    app.add_routes(routes)
    return app


if __name__ == "__main__":
    web.run_app(init_app())
