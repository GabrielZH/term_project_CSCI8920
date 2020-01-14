import os

count = 0

try:
    while count < int(1e4):
        os.system("python3 -m custom_agent"
                  " --agent mbrl_agent_repeated_single_game.MdpModelBasedAgent"
                  " --map CollectMineralShards"
                  " --use_feature_units true"
                  " --max_episodes 1")
except KeyboardInterrupt:
    raise
