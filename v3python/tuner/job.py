#!/usr/bin/env python

from pathlib import Path
import json
import uuid
import sqlite3
from .util import (
    get_sql
)

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
    SQL_ATOMIC_ASSIGN = get_sql('atomic_assign')

    def __init__(self, path : Path):
        super().__init__(path)
        dbf = self._dir / self.DBFILE
        if not dbf.is_file():
            raise RuntimeError(f"Job database {dbf} does not exist")
        self._con = sqlite3.connect(dbf)

    def assign(*, count, worker, remote_addr):
        con.execute(self.SQL_ATOMIC_ASSIGN, (worker, remote_addr, count))
        return con.fetchall()

class NewJob(BaseJob):
    SQL_INIT_TABLE = get_sql('init_table')
    SQL_ADD_TASKS = get_sql('add_tasks')

    def __init__(self,
                 info : dict,
                 workdir : Path):
        super().__init__(workdir / uuid.uuid4())
        con = sqlite3.connect(self._dir / self.DBFILE)
        self._init_table(con, info)

    @staticmethod
    def _init_table(con : sqlite3.Connection, info : dict):
        con.execute(SQL_INIT_TABLE)
        tasks = info['tasks']
        def gen():
            for i, task in enumerate(tasks):
                yield i, json.dumps(task)
        rows = [ tup for tup in gen() ]
        con.executemany('UPDATE Primary SET id = ?, jdesc = ?', rows)
        con.execute('UPDATE Primary SET status = ?, client = ?, ip = ?', (PrimaryStatus.UNASSIGNED, '', ''))
