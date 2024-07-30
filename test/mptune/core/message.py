#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from enum import Enum
from copy import deepcopy

MonadAction = Enum('MonadAction', ['Pass',
    'Skip',
    'DryRun',
    'Exit',
    'Exception',
    'OOB_Init',
    'OOB_RequestStatus',  # OOB means side channel communication only
    'OOB_AckRecv',
])

'''
Message passed through multiprocessing.Queue

Nothing special but needs an ID
'''
class MonadMessage(ABC):

    def __init__(self, *, task_id, action : MonadAction, source=None, payload=None):
        self._task_id = task_id
        self._action = action
        self._source = source
        self._payload = payload

    def __format__(self, format_spec):
        return f'MonadMessage(task_id={self.task_id}, action={self.action}, source={self.source}, payload={self.payload})'

    @property
    def task_id(self):
        return self._task_id

    @property
    def action(self):
        return self._action

    @property
    def source(self):
        return self._source

    @property
    def payload(self):
        return self._payload

    @property
    def skip_reason(self):
        if hasattr(self, '_skip_reason'):
            return self._skip_reason
        return None

    def set_source(self, source):
        self._source = source
        return self

    def update_payload(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self._payload, k, v)
        return self

    def make_skip(self, monad, reason=None) -> 'MonadMessage':
        ret = self.forward(monad)
        ret._skip_reason = reason
        return ret

    def make_dryrun(self, monad) -> 'MonadMessage':
        ret = deepcopy(self)
        ret.set_source = monad.identifier
        ret._action = MonadAction.DryRun
        return ret

    # def make_pass(self, **kwargs) -> 'MonadMessage':
    #     ret = deepcopy(self)
    #     ret._action = MonadAction.Pass
    #     for k, v in kwargs.items():
    #         setattr(ret, k, v)
    #     return ret

    def clone_ackrecv(self, monad) -> 'MonadMessage':
        ret = deepcopy(self).set_source(monad.identifier)
        ret._action = MonadAction.OOB_AckRecv
        return ret

    def forward(self, monad) -> 'MonadMessage':
        ret = deepcopy(self).set_source(monad.identifier)
        return ret

# class QueuePair(object):
# 
#     def __init__(self):
#         self._request_flow = None
#         self._feedback_flow = None
#         self._monads_up = set()
#         self._monads_down = set()
# 
#     @property
#     def request_flow(self):
#         if self._request_flow is None:
#             self._request_flow = Queue()
#         return self._request_flow
# 
#     @property
#     def feedback_flow(self):
#         if self._feedback_flow is None:
#             self._feedback_flow = Queue()
#         return self._feedback_flow
# 
#     def connect(self, up : Monad, down : Monad):
#         self._monads_up.add(up)
#         self._monads_down.add(down)
