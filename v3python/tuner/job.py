#!/usr/bin/env python

from pathlib import Path
import json
import uuid
import sqlite3

''' markdown
# Table

## Primary

* id : int, primary key
* desc: text, json string to completely describe the tuning task
* status: int as enum, class Status
* assignment: text, which node this task is assigned to.
* ip: text, remote ip address
* deadline: DATE, when the ASSIGNED task is considered as timed out

## Secondary

* primary : int, primary key, foreign key Primary(id)
* kernel_name : TEXT, primary key
* kernel_index : int, primary key
* tune_status : int as enum, class TuneStatus

'''

class PrimaryStatus:
    IDLE = 0
    ASSIGNED = 1
    COMPLETE = 2
    TIMEOUT = 3

class SecondaryStatus:
    UNKNOWN = 0
    ACK = 1
    NAK = 2
    CAN = 3

class BaseJob(object):
    DBFILE = 'job.sqlite3'

    def __init__(self, path : Path):
        self._dir = path

class Job(BaseJob):
    def __init__(self, path : Path):
        super().__init__(path)
        dbf = self._dir / self.DBFILE
        if not dbf.is_file():
            raise RuntimeError(f"Job database {dbf} does not exist")
        self._con = sqlite3.connect(dbf)

    '''
    UPDATE Primary SET
        status = ASSIGNED,
        assignment = {worker},
        ip = {remote_addr}
    WHERE
        status = IDLE,
    RETURNING *
    LIMIT {count}
    '''
    def assign(*, count, worker, remote_addr):
        pass

class NewJob(BaseJob):
    def __init__(self,
                 info : dict,
                 workdir : Path):
        super().__init__(workdir / uuid.uuid4())
        con = sqlite3.connect(self._dir / self.DBFILE)
        self._init_table(con, info)

    @staticmethod
    def _init_table(con : sqlite3.Connection, info : dict):
        tasks = info['tasks']
        def gen():
            for i, task in enumerate(tasks):
                yield i, json.dumps(task)
        values = [ tup for tup in gen() ]
        con.executemany('UPDATE Primary SET status = ?, assignment = ?', (Status.UNASSIGNED, ''))
        con.execute('UPDATE Primary SET status = ?, assignment = ?', (Status.UNASSIGNED, ''))
