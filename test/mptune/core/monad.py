#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from multiprocessing import Process, Queue
import queue
from collections import namedtuple
from copy import deepcopy
from .aav import ArgArchVerbose
from .message import MonadAction, MonadMessage

'''
A Monad will runs in a separate process.
For the main process this is a proxy object.

This class is supposed to be inherited to actual implement a service in cpptuning framework.
'''
class Monad(ArgArchVerbose):

    def __init__(self, args, identifier=None, side_channel=None, init_object=None):
        super().__init__(args)
        self._q_up = None
        self._q_down = None
        self._identifier = identifier  # None identifier implies unique in the system
        self._process = None
        self._side_channel = side_channel
        self._init_object = init_object

    @property
    def identifier(self):
        return self._identifier

    '''
    1-1 Protocol: assert to ensure only creating QP once
    '''
    def bind_1to1(self, downstream : 'Monad'):
        assert self._q_down is None
        assert downstream._q_up is None
        self._q_down = downstream._q_up = Queue()
        self._q_down.cancel_join_thread()
        # self._q_down.connect(self, downstream)

    '''
    1-N Protocol: assert to ensure no overwriting
    '''
    def bind_1toN(self, downstreams : 'Monad'):
        assert self._q_down is None
        q = Queue()
        q.cancel_join_thread()
        self._q_down = q
        for ds in downstreams:
            assert ds._q_up is None
            ds._q_up = q
            # self._q_down.connect(self, ds)

    '''
    N-1 Protocol: reuse downstream QP
    '''
    def bind_Nto1(self, downstream : 'Monad'):
        if downstream._q_up is None:
            downstream._q_up = Queue()
            downstream._q_up.cancel_join_thread()
        assert self._q_down is None
        self._q_down = downstream._q_up
        # self._q_down.connect(self, downstream)

    # def plain_start(self, init_object):
    #     assert self._q_up is None
    #     self.init_obj = None
    #     self._process = Process(target=self.main, args=(self._q_up, self._q_down, None))

    def start(self):
        self._process = Process(target=self.main, args=(self._q_up, self._q_down, None))
        self._process.start()

    def join(self):
        self._process.join()

    '''
    Join, and then start a new process
    '''
    def restart_with_last_progress(self, continue_from):
        self._process.join()
        self._process = Process(target=self.main, args=(self._q_up, self._q_down, continue_from))
        self._process.start()

    @property
    def sentinel(self):
        return self._process.sentinel

    @property
    def exitcode(self):
        return self._process.exitcode

    def main(self, q_up, q_down, continue_from):
        service = self.service_factory()
        # if q_up:
        #     if hasattr(self, '_init_obj_for_restart'):
        #         init_obj = self._init_obj_for_restart
        #     else:
        #         init_obj = q_up.get()
        #         self._init_obj_for_restart = init_obj
        service.init(self._init_object)
        if self._side_channel:
            self._side_channel.put(MonadMessage(task_id=None, action=MonadAction.OOB_Init, source=self.identifier))
        self.main_loop(service, q_up, q_down, continue_from)
        service.cleanup()

    def main_loop(self, service, q_up, q_down, continue_from):
        while True:
            try:
                if q_up is None:
                    req = None
                else:
                    if continue_from is not None:
                        req = continue_from
                        continue_from = None
                    elif q_up:
                        req = q_up.get()
                        print(f'Monad {self.identifier} receives {req}')
                        if req.action == MonadAction.Exit:
                            '''
                            Exit handling: instead of broadcasting in a 1-N queue,
                            putting the object back to the shared queue.
                            '''
                            # q_up.put(req)
                            pass
                    if self._side_channel:
                        self._side_channel.put(req.clone_ackrecv(self))
                '''
                Note for MonadService.process: propogate Exit to downstream.
                     Unless not needed.
                '''
                for res in service.process(req):
                    res.set_source(self._identifier)
                    '''
                    Report to downstream and/or side channel
                      - downstream for further processing
                      - side channel for status tracking
                    '''
                    if self._identifier == 'StateTracker':
                        print(f'StateTracker yield {res}')
                    if q_down:
                        q_down.put(res)
                    if self._side_channel:
                        self._side_channel.put(res.forward(self))
                    if res.action == MonadAction.Exit:
                        print(f'Monad {self.identifier} exits the loop')
                        return
            except ValueError:  # mp.Queue closed
                break

    @abstractmethod
    def service_factory(self):
        pass

    def report_exception(self, e: Exception):
        msg = MonadMessage(task_id=None, action=MonadAction.Exception)
        msg.exception = e
        self._side_channel.put(msg)

    def report_status(self, msg):
        self._side_channel.put(msg)

class MonadService(ArgArchVerbose):

    def __init__(self, args, monad):
        super().__init__(args)
        self._monad = monad  # In case need it

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def process(self, request):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    @property
    def monad(self):
        return self._monad

    def report_exception(self, e: Exception):
        self._monad.report_exception(e)

    def report_status(self, msg):
        self._monad.report_status(msg)
