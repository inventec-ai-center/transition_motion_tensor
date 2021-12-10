import numpy as np
import pickle as pkl
from env.env import Env
from deepmimic_ppo_agent import DeepMimicPPOAgent
import random


class TMTDeepMimicAgent(DeepMimicPPOAgent):
    NAME = "TMT_DeepMimic_Agent"

    Transition_Phase_Threshold = 2e-3
    Force_Update_Action_After_Switch_Motion = True
    Transition_Expired_Time = 2

    def __init__(self, world, id, json_data, seed):
        self.transition_queue = []
        self.motion_prev = ''
        self.prev_switch_motion_time = 0
        self.transition_queue_update_time = 0
        self.transition_tensor = None
        super().__init__(world, id, json_data, seed)
        return

    def load_transition_tensor(self, tmt_file):
        self.transition_tensor = pkl.load(open(tmt_file, "rb"))
        return

    def reset(self):
        self.transition_queue.clear()
        super().reset()
        return

    def update(self, timestep):
        self._update_transition_queue(timestep)
        super().update(timestep)
        return

    def _update_transition_queue(self, timestep):
        wall_time = self._get_time()

        if len(self.transition_queue) > 0:
            src_phase = self._record_phase()
            head = self.transition_queue[0]
            target_src_phase, target_dst_phase, src_motion, dst_motion = head['src_phase'], head['dst_phase'], head['src_motion'], head['dst_motion']
            # print("Transition Queue Head: {src_motion: %s, dst_motion: %s, src_phase: %.3f, dst_phase: %.3f, wall_time: %.3f}, wall_time: %.3f, src_phase: %.3f" % \
            #     (src_motion, dst_motion, target_src_phase, target_dst_phase, head['wall_time'], wall_time, src_phase))

            # delete expired transitions
            if wall_time - head['wall_time'] > self.Transition_Expired_Time:
                self.transition_queue = self.transition_queue[1:]
                print('pop expired transitions from queue')
                return

            if np.abs(src_phase - target_src_phase) < self.Transition_Phase_Threshold:
                print("Switch motion {}: {:.3f} - {}: {:.3f} (wall_time={:.6f})\n".format(src_motion, src_phase, dst_motion, target_dst_phase, self._get_time()))
                dst_motion_index = self.motion_names.index(dst_motion)
                _ = self.switch_motion(src_motion, dst_motion_index, target_dst_phase)
                self.transition_queue = self.transition_queue[1:]
        return

    def keyboard(self, key):
        src_motion_idx = self._record_motion_label()
        src_motion = self.motion_names[src_motion_idx]
        src_phase = self._record_phase()

        ret = None
        key_char = key.decode("utf-8")
        if key_char >= '1' and key_char <= '9':
            dst_motion_index = int(key_char) - 1
            if src_motion_idx != dst_motion_index and dst_motion_index < len(self.motion_names):
                ret = self.query_transition(src_motion, src_phase, self.motion_names[dst_motion_index])

        if ret:
            target_src_phase, target_dst_phase, dst_motion = ret['src_phase'], ret['dst_phase'], ret['dst_motion']
            if np.abs(src_phase - target_src_phase) < self.Transition_Phase_Threshold:
                print("Switch motion {}: {:.3f} - {}: {:.3f}\n".format(src_motion, src_phase, dst_motion, target_dst_phase))
                dst_motion_index = self.motion_names.index(dst_motion)
                _ = self.switch_motion(src_motion, dst_motion_index, target_dst_phase)
            else:
                self.transition_queue.append(ret)
                print("Transition {}: {:.3f} - {}: {:.3f} is added to transiton queue".format(src_motion, target_src_phase, dst_motion, target_dst_phase))
        return

    def query_transition(self, src_motion, src_phase, dst_motion):
        if self.transition_tensor is None: return None
        if len(self.transition_queue) > 0 and self.transition_queue[0]['src_motion'] == src_motion and self.transition_queue[0]['dst_motion'] == dst_motion: return None

        transition_key = "{}_{}".format(src_motion, dst_motion)
        if transition_key not in self.transition_tensor or len(self.transition_tensor[transition_key]) == 0:
            return None

        # Look for the source phase
        osrc_phases = []
        osrc_phase_idxs = []
        for idx, tup in enumerate(self.transition_tensor[transition_key]):
            osrc_phases.append(tup[0])
            osrc_phase_idxs.append(idx)

        if len(osrc_phases) == 0:
            return None

        osrc_phases = np.array(osrc_phases, dtype=np.float32)
        osrc_phases = np.concatenate([osrc_phases, osrc_phases + 1])
        _phase_indices = np.where((osrc_phases - src_phase) > 0)[0] % len(osrc_phase_idxs)
        if len(_phase_indices) == 0:
            return None
        src_phase_idx = osrc_phase_idxs[_phase_indices[0]]  # Use the closest next phase
        ret = self.transition_tensor[transition_key][src_phase_idx]
        osrc_phase, odst_phase = ret[0], ret[1]

        if osrc_phase >= 0:
            out_dict = {
                "src_motion": src_motion,
                "dst_motion": dst_motion,
                "src_phase": osrc_phase,
                "dst_phase": odst_phase,
                "wall_time": self._get_time()
            }
            return out_dict
        return None

    def switch_motion(self, src_motion, dst_motion, dst_phase):
        wall_time = self._get_time()
        self.motion_prev = src_motion
        self.prev_switch_motion_time = wall_time
        succ = self._switch_motion(dst_motion, float(dst_phase))
        if self.Force_Update_Action_After_Switch_Motion:
            self._update_new_action()
        return succ

    def _record_phase(self):
        motion_label = self.world.env.record_phase(self.id)
        return motion_label

    def _switch_motion(self, target_motion, target_phase):
        succ = self.world.env.switch_motion(self.id, target_motion, target_phase)
        return succ

    def _get_time(self):
        return self.world.env.get_time()

    def isDone(self):
        return False