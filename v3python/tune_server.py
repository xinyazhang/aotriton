#!/usr/bin/env python

import os
import json

WORKDIR = int(os.getenv('AOTRITON_TUNER_WORKDIR', default=None))
assert WORKDIR is not None, 'Must set environment variable AOTRITON_TUNER_WORKDIR to launch tuner server'

from .tuner import (
    Server,
)
Server.set_workdir(WORKDIR)

from flask import (
    Flask,
    request,
    abort,
    make_response,
)

app = Flask(__name__)

@app.get("/job/<uuid>")
def job_get(uuid):
    count = int(request.args.get('count', '1'))
    client = request.args.get('client', '')
    if not client:
        abort(make_response("Need 'client' argument to identify itself", 400))
    job = Server.locate(uuid)
    if job is None:
        abort(make_response(f"Unknown Job UUID {uuid}", 400))
    return job.assign(count=count, worker=client, remote_addr=request.remote_addr)

@app.get("/job/<uuid>")
def job_post(uuid):
    job = Server.locate(uuid)
    j = request.get_json()
    return job.update(j)

# TODO: @app.get("/watchdog/<uuid>")

@app.post("/admin/add_job")
def job_post():
    if not request.remote_addr.startswith('127.'):
        abort(403)
    file = request.files['json_file']
    try:
        j = json.load(file)
    except:
        abort(make_response(f"Invalid JSON document", 400))
    try:
        return Server.create_job(info = j)
    except Exception as e:
        abort(make_response("Exception when creating job: " + str(e), 400))
