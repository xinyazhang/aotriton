#!/usr/bin/env python

from pathlib import Path
from .job import (
    Job,
    NewJob,
)


class Server(object):
    WORKDIR = None
    CACHE = {}

    @staticmethod
    def set_workdir(work_dir):
        Server.WORKDIR = Path(work_dir)

    @staticmethod
    def locate(uuid):
        if uuid in Server.CACHE:
            return Server.CACHE
        p = Server.WORKDIR / uuid
        if not p.is_dir():
            # Do not update cache. the job can be created later
            return None
        ret = Job(p)
        Server.CACHE[uuid] = ret
        return ret

    @staticmethod
    def create_job(*, info : dict) -> dict:
        job = NewJob(info : dict, workdir=Server.WORKDIR)
        return { 'success' : job.is_valid(), 'uuid' : job.uuid }
