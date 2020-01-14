Environment requirements: https://github.com/deepmind/pysc2
(Note) The pip version of PySC2 has been outdated, it should be installed from source.

---------------------------------------------------------------------------------------

                               -Introduction of Source Files-
mbrl_agent_repeated_single_game.py -- training an MDP model-based agent on a single game repetitiously.
mbrl_agent_random_game.py -- training an MDP model-based agent on randomly reset games.
mdp_model.py -- definition of the MDP model.
evaluation.py -- test the performance of a trained agent with the police it finally optimized.

running_agent.py -- a script controlling the number of iteration of training an agent on repeated single game.
custom_agent.py -- subclass of pysc2.bin.agent using to get the initial state fixed, which ensures a repeated game.
utils -- some tool functions needed in other major source files.

base_agent -- parent class of all implemented agents, which defines the interaction between environment and agents.
random_agent -- an agent taking ranomd actions at each agent step.
scripted_agent -- an agent taking actions based on manually-assigned strategy.

---------------------------------------------------------------------------------------

                                        -How to Run-

1. Go under linux console.
2. Switch to the current directory.
3. Type command:
   (For training an agent for repeated single game)
   python3 -m custom_agent --agent mbrl_agent_repeated_single_game.MdpModelBasedAgent --map CollectMineralShards --use_feature_units true --max_episodes (an INTEGER, e.g. 10000)

   (For training an agent for random reset games)
   python3 -m pysc2.bin.agent --agent mbrl_agent_random_game.MdpModelBasedAgent --map CollectMineralShards --use_feature_units true --max_episodes (an INTEGER)

   (For evaluating an agent)
   python3 -m custom_agent/pysc2.bin.agent --agent evaluation.EvaModelBasedAgent --map CollectMineralShards --use_feature_units true --max_episodes (an INTEGER)

   (For the scripted agent)
   python3 -m pysc2.bin.agent --agent pysc2.agents.scripted_agent.CollectMineralShardsFeatureUnits --map CollectMineralShards --use_feature_units true --max_episodes (an INTEGER) 

---------------------------------------------------------------------------------------
* All the code need to be compiled under required environments.
* The final policy written in a text file is too big to submit.
