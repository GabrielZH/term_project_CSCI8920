from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pysc2.agents import base_agent
from pysc2.lib import actions, features

import mbrl_agent

from utils import fs2dict, dict2fs, list2tuple, tuple2list


_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

concise_act_ids = (2, 3, 4, 5, 7, 12, 13, 331, 332)


class EvaModelBasedAgent(base_agent.BaseAgent):
    # _policy = open('policy_sg.txt', 'rt').read()
    _policy = open('policy_rg.txt', 'rt').read()
    policy = eval(_policy)

    def step(self, obs):
        super(EvaModelBasedAgent, self).step(obs)

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

        state = {"marines": marines, "minerals": minerals}

        if dict2fs(state) in self.policy:
            func_id = self.policy[dict2fs(state)][0]
            func_args = tuple2list(self.policy[dict2fs(state)][1])
            act = actions.FunctionCall(func_id, func_args)
            return act
        else:
            function_id = 0
            while function_id not in concise_act_ids:
                function_id = np.random.choice(obs.observation.available_actions)
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[function_id].args]
            act = actions.FunctionCall(function_id, args)
            return act
