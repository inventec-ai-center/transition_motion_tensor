import json
import numpy as np
from learning.ppo_agent import PPOAgent
from carl_ppo_agent import CarlPPOAgent
from tmt_carl_agent import TMTCarlAgent
from tmt_deepmimic_agent import TMTDeepMimicAgent
from deepmimic_ppo_agent import DeepMimicPPOAgent

import sys
sys.path.append('../DeepMimic/')
import learning.agent_builder as DMAgentBuilder

AGENT_TYPE_KEY = "AgentType"

def build_agent(world, id, file, seed=None):
    agent = None
    with open(file) as data_file:
        json_data = json.load(data_file)

        assert AGENT_TYPE_KEY in json_data
        agent_type = json_data[AGENT_TYPE_KEY]

        if agent_type == CarlPPOAgent.NAME:
            agent = CarlPPOAgent(world, id, json_data, seed)
        elif agent_type == DeepMimicPPOAgent.NAME:
            agent = DeepMimicPPOAgent(world, id, json_data, seed)
        elif agent_type == TMTCarlAgent.NAME:
            agent = TMTCarlAgent(world, id, json_data, seed)
        elif agent_type == TMTDeepMimicAgent.NAME:
            agent = TMTDeepMimicAgent(world, id, json_data, seed)
        else:
            agent = DMAgentBuilder.build_agent(world, id, file)

    return agent
