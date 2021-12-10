import numpy as np
import agent_builder as AgentBuilder
import learning.tf_util as TFUtil
from learning.rl_agent import RLAgent
from tmt_carl_agent import TMTCarlAgent
from tmt_deepmimic_agent import TMTDeepMimicAgent
from util.logger import Logger
import learning.rl_world as DeepMimicRLWorld


class RLWorld(DeepMimicRLWorld.RLWorld):

    def __init__(self, env, arg_parser):
        TFUtil.disable_gpu()

        self.env = env
        self.arg_parser = arg_parser
        self._enable_training = True
        self.train_agents = []
        self.parse_args(arg_parser)

        self.build_agents()
        return

    def build_agents(self):
        num_agents = self.env.get_num_agents()
        self.agents = []

        Logger.print('')
        Logger.print('Num Agents: {:d}'.format(num_agents))

        agent_files = self.arg_parser.parse_strings('agent_files')
        assert(len(agent_files) == num_agents or len(agent_files) == 0)

        model_files = self.arg_parser.parse_strings('model_files')
        motion_names = self.arg_parser.parse_strings('motion_names')
        primitive_models = self.arg_parser.parse_strings('primitive_models')
        gating_models = self.arg_parser.parse_strings('gating_models')
        normalizer_file = self.arg_parser.parse_string('normalizer_file')
        transition_tensor_file = self.arg_parser.parse_string('transition_tensor_file')

        output_path = self.arg_parser.parse_string('output_path')
        int_output_path = self.arg_parser.parse_string('int_output_path')

        global_rand_seed = self.arg_parser.parse_int('rand_seed')

        for i in range(num_agents):
            curr_file = agent_files[i]
            curr_agent = self._build_agent(i, curr_file, global_rand_seed)

            if curr_agent is not None:
                curr_agent.output_dir = output_path
                curr_agent.int_output_dir = int_output_path
                Logger.print(str(curr_agent))

                if len(model_files) == len(motion_names):
                    curr_agent.load_actor_models(motion_names, model_files)
                elif len(gating_models) == len(primitive_models) == len(motion_names):
                    for motion_name, primitive_model_path, gating_model_path in zip(motion_names, primitive_models, gating_models):
                        curr_agent.load_primitive_model(motion_name, primitive_model_path)
                        curr_agent.load_gating_model(motion_name, gating_model_path)

                if normalizer_file:
                    curr_agent.load_normalizer(normalizer_file)

                if transition_tensor_file and (type(curr_agent) == TMTCarlAgent or type(curr_agent) == TMTDeepMimicAgent):
                    curr_agent.load_transition_tensor(transition_tensor_file)

            self.agents.append(curr_agent)
            Logger.print('')

        self.set_enable_training(self.enable_training)
        return

    def _build_agent(self, id, agent_file, seed):
        Logger.print('Agent {:d}: {}'.format(id, agent_file))
        if agent_file == 'none':
            agent = None
        else:
            agent = AgentBuilder.build_agent(self, id, agent_file, seed)
            assert (agent != None), 'Failed to build agent {:d} from: {}'.format(id, agent_file)

        return agent

    def isDone(self):
        isDone = False
        for agent in self.agents:
            if (agent is not None):
                isDone |= agent.isDone()
        return isDone

    def keyboard(self, key, x, y):
        key_val = int.from_bytes(key, byteorder='big')
        self.env.keyboard(key_val, x, y)

        for agent in self.agents:
            if type(agent) == TMTCarlAgent or type(agent) == TMTDeepMimicAgent:
                agent.keyboard(key)
        return
