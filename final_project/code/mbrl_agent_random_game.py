from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os

from pysc2.agents import base_agent
from pysc2.lib import actions, features

import mdp_model
from utils import fs2dict, dict2fs, list2tuple, tuple2list


_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

concise_act_ids = (2, 3, 4, 5, 7, 12, 13, 331, 332)


class MdpModelBasedAgent(base_agent.BaseAgent):
    mdp = mdp_model.MarkovDecisionProcess(discount=0.9, prior_count=1)
    states = mdp.states
    acts = mdp.acts
    # trans_model = mdp.MarkovDecisionProcess.trans_model
    gamma = mdp.DISCOUNT
    c = mdp.PRIOR_COUNT

    act_vals = dict()
    rwds = dict()
    trans_count = dict()

    policy = dict()
    state = {"marines": None, "minerals": None}
    act = actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    agent_step = 0

    def step(self, obs):
        self.agent_step += 1

        super(MdpModelBasedAgent, self).step(obs)
        marine_units = [unit for unit in obs.observation.feature_units
                        if unit.alliance == _PLAYER_SELF]
        marines = frozenset(
            {dict2fs(
                {"coord": (marine_units[0].x, marine_units[0].y),
                 "selected": marine_units[0].is_selected}
            ),
                dict2fs(
                    {"coord": (marine_units[1].x, marine_units[1].y),
                     "selected": marine_units[1].is_selected}
                )}
        )
        minerals = frozenset([(unit.x, unit.y)
                              for unit in obs.observation.feature_units
                              if unit.alliance == _PLAYER_NEUTRAL])
        next_state = {"marines": marines, "minerals": minerals}
        if next_state == self.state:
            function_id = np.random.choice(obs.observation.available_actions)
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            self.act = actions.FunctionCall(function_id, args)
            return self.act

        cur_rwd = obs.reward

        if next_state not in self.states:
            self.states.append(next_state)

        # if self.act not in self.acts:
        #     self.acts.append(self.act)

        trans = (dict2fs(self.state),
                 dict2fs(next_state),
                 (self.act.function, list2tuple(self.act.arguments)))
        if trans in self.trans_count:
            self.trans_count[trans] += 1
        else:
            self.trans_count[trans] = 1
            # self.trans_model[trans] = 0

        if trans not in self.rwds:
            self.rwds[trans] = cur_rwd
        self.rwds[trans] += (cur_rwd - self.rwds[trans]) \
                            / self.trans_count[trans]

        self.state = next_state

        self.act_vals[(dict2fs(self.state),
                       (self.act.function,
                        list2tuple(self.act.arguments)))] = cur_rwd

        # async_iter_round = len(self.trans_count)
        # if async_iter_round > 1e3:
        #     async_iter_round = 1e3
        # iter_count = 0
        # while iter_count < async_iter_round:
        self.policy, self.act_vals = self.async_val_iter()
            # iter_count += 1

        if self.agent_step >= 96000:
            # _policy = dict()
            # for st in self.policy:
            #     _policy[str(st)] = self.policy[st]
            _policy = str(self.policy)
            f = open("policy_random_game.txt", "w")
            f.write(_policy)
            f.close()

        if dict2fs(self.state) in self.policy:
            func_id = self.policy[dict2fs(self.state)][0]
            func_args = tuple2list(self.policy[dict2fs(self.state)][1])
            self.act = actions.FunctionCall(func_id, func_args)
            return self.act
        else:
            function_id = 0
            while function_id not in concise_act_ids:
                function_id = np.random.choice(
                    obs.observation.available_actions)
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            self.act = actions.FunctionCall(function_id, args)
            return self.act

    def async_val_iter(self):
        rand_idx = np.random.choice(len(self.states))
        state = self.states[rand_idx]
        act_vals = self.act_vals
        policy = self.policy
        max_actval4st = 0.0
        trans_ct = 0
        tmp_trans_store = list()
        for trans in self.trans_count:
            if dict2fs(state) == trans[0]:
                act = trans[2]
                for t in self.trans_count:
                    if dict2fs(state) == t[0] and act == t[2]:
                        trans_ct += (self.trans_count[t] + self.c)
                        tmp_trans_store.append(t)
                updated_act_val = 0.0
                for t in tmp_trans_store:
                    trans_prob = (self.trans_count[t] + self.c) / trans_ct
                    tmp_st_act_store = dict()
                    for st_act in act_vals:
                        if st_act[0] == t[1]:
                            tmp_st_act_store[st_act] = act_vals[st_act]
                    if len(tmp_st_act_store) <= 0:
                        continue

                    max_next_st_act_val = tmp_st_act_store[
                        max(tmp_st_act_store)]
                    # print(
                    #     "prob: ", trans_prob,
                    #     "trans_reward: ", self.rwds[t],
                    #     "max_next_act_val: ", max_next_st_act_val
                    # )
                    updated_act_val += trans_prob \
                                       * (self.rwds[t]
                                          + self.gamma * max_next_st_act_val)
                    act_vals[(dict2fs(state), act)] = updated_act_val

                if updated_act_val > max_actval4st:
                    policy[dict2fs(state)] = act
                    max_actval4st = act_vals[(dict2fs(state), act)]

        return policy, act_vals
