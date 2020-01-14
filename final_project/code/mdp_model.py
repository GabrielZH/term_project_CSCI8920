from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class MarkovDecisionProcess:

    def __init__(self, discount=1.0, prior_count=0):
        self.states = list()
        self.acts = list()
        self.trans_model = dict()
        self.DISCOUNT = discount
        self.PRIOR_COUNT = prior_count

    def get_states(self):
        return self.states

    def get_init_state(self):
        pass

    def get_trans_model(self):
        return self.trans_model

    def get_next_state(self, cur_state, act):
        pass
