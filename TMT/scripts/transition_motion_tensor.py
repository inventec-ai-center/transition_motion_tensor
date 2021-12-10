import os
import numpy as np
import pandas as pd
import json
import pickle as pkl
from tqdm import tqdm
from tmt_util import get_bin, bin2int, find_cycles_from_contact, global2local, load_json

class TransitionMotionTensor:
    _columns = ["src_motion","src_phase","dst_motion","dst_phase","dst_gphase",     # Transition Indices
                "is_alive", "effort", "duration", "control_accuracy",               # Transition Outcomes
                "during_pos","post_pos"]                                            # Clustering Purposes

    # Each motion's duration in seconds
    _cycle_duration = {"Pace": 0.6445, "Trot": 0.4844, "Canter": 0.4381, "Jump": 0.9335, # dog
                       "Walk": 1.266616, "Jog": 0.833333, "Run": 0.799968} # humanoid

    def __init__(self, tensor_name, cache_dir="./cache"):
        self._tensor_name = tensor_name
        self._cache_dir = cache_dir
        self._samples = []
        self._motions = []
        self._data = None
        self._pdata = None

    def _process_sample(self, sample, recording_fps=60):
        if sample["src_phase"] == 0: return None
        if sample['src_motion'] not in self._cycle_duration.keys(): return None
        if sample['dst_motion'] not in self._cycle_duration.keys(): return None

        is_alive = int(sample["is_fallen"] == False)  # Transition Outcome

        # Get the start index for the transition
        start = sample["transition_index"]

        # Set a default value for the effort, duration, and control accuracy
        effort, duration, control_accuracy = (np.nan, np.nan, np.nan)
        during_pos, post_pos = (np.nan, np.nan)

        if is_alive:
            # Extract the cycles
            foot_contacts = np.array([bin2int(x) for x in sample["foot_contacts"]])
            heights = [p[1] for p in sample['position']]
            control_reward = (np.array(sample["speed_reward"]) + np.array(sample["heading_reward"]) + np.array(sample["height_reward"]))/3
            cycles = find_cycles_from_contact(foot_contacts[start:], heights[start:], control_reward[start:], sample["dst_motion"]) + start

            # Ignore this particular sample due to insufficient cycles, potentially noisy recording
            if len(cycles) < 4: return None
            end = cycles[0]
            duration = end - start

            # Compute energy consumption relative to the destination
            transition_energy = np.sum(sample["loss_energy"][start:end])
            energy_list = []
            for i in range(len(cycles)-1):
                st_point1 = cycles[i]
                st_point2 = cycles[i+1]
                energy = np.sum(sample["loss_energy"][st_point1:st_point2])
                energy_list.append(energy)
            stable_energy = np.mean(energy_list)
            effort = transition_energy/stable_energy

            # Control accuracy: The sum of control reward between the first and second stable states
            control_accuracy = np.sum(control_reward[cycles[0]:cycles[1]])

            # Other useful information used for clustering
            pos = np.array(sample["position"])
            during_pos = pos[start:end]
            post_pos = pos[cycles[0]: cycles[1]]

        # In the recording, we use the global_kin_char phase as the destination.
        # For the tensor, we need to convert that to each motion cycle (local phase).
        dst_phase = global2local(sample["dst_phase"], sample["dst_motion"])
        sample_tuple = (sample["src_motion"], sample["src_phase"], sample["dst_motion"], dst_phase, sample["dst_phase"],
                        is_alive, duration, effort, control_accuracy,
                        during_pos, post_pos)
        return sample_tuple

    def add_sample(self, sample_tuple):
        assert len(self._columns) == len(sample_tuple)
        self._samples.append(sample_tuple)

    def load_from_cache(self, sample_cache_file, quality_cache_file):
        if os.path.exists(sample_cache_file):
            print("Found cache! Loading from {}".format(sample_cache_file))
            self._data = pkl.load(open(sample_cache_file, "rb"))

            # get available motions within samples
            self._motions = []
            for motion in list(self._cycle_duration.keys()):
                if (self._data[(self._data.src_motion == motion)].shape[0] > 0 and
                    self._data[(self._data.dst_motion == motion)].shape[0] > 0):
                    self._motions.append(motion)
            print('Available motions:', self._motions)

        if os.path.exists(quality_cache_file):
            print("Found cache! Loading from {}".format(quality_cache_file))
            self._pdata = pkl.load(open(quality_cache_file, "rb"))

    def load_from_json(self, trajectory_paths, recording_fps=60):
        cache_name = "{}/{}-JSON-NTrajectory{}.pkl".format(self._cache_dir, self._tensor_name, len(trajectory_paths))
        if os.path.exists(cache_name):
            print("Found cache! Loading from {}".format(cache_name))
            self._data = pkl.load(open(cache_name, "rb"))
        else:
            # If cache doesn't exist
            for path in tqdm(trajectory_paths):
                samples = load_json(path)
                if samples is None: continue

                # Iterate through the samples in the trajectory
                for sample in samples:
                    if sample['is_valid']:
                        sample_tuple = self._process_sample(sample)

                        # Ignore invalid samples
                        if sample_tuple:
                            self.add_sample(sample_tuple)

            # Once all samples are added build the DataFrame
            self._samples = np.stack(self._samples)
            self._data = pd.DataFrame(columns=self._columns, data=self._samples)

            for col in self._columns:
                if col in ["src_motion","dst_motion","during_pos","post_pos"]: continue
                self._data[col] = self._data[col].astype("float32")

            # save cache
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
            pkl.dump(self._data, open(cache_name, "wb"))
            print('Saved cache to', cache_name)

        # get available motions within samples
        self._motions = []
        for motion in list(self._cycle_duration.keys()):
            if (self._data[(self._data.src_motion == motion)].shape[0] > 0 and
                self._data[(self._data.dst_motion == motion)].shape[0] > 0):
                self._motions.append(motion)
        print('Available motions:', self._motions)

    def _compute_stability(self, df, w, dt):
        '''
        Important! df, should contain only one pair of source and destination motion.
        dt: a tolerance variable in seconds
        '''
        pdf = df.copy(deep=True)

        # Convert seconds to phase
        src_motion = pdf.src_motion.unique()
        dst_motion = pdf.dst_motion.unique()
        assert (len(src_motion) == 1) & (len(dst_motion) == 1)
        src_motion = src_motion[0]
        dst_motion = dst_motion[0]

        dt_i = dt / self._cycle_duration[src_motion]
        dt_j = dt / self._cycle_duration[dst_motion]

        # Evaluate local neighbors
        st_outcome = []
        st_alive = []

        for itr, row in tqdm(pdf.iterrows(), total=pdf.shape[0]):
            pi = row.src_phase
            pj = row.dst_phase

            # Local neighborhood
            subtensor = df[df.src_phase.between(pi-dt_i, pi+dt_i) &
                           df.dst_phase.between(pj-dt_j, pj+dt_j)]

            st_alive.append(subtensor.is_alive.sum() / subtensor.shape[0])
            st_outcome.append(np.var(subtensor.outcomes))

        pdf.loc[:, "st_alive"] = st_alive  # Alive proportion around the sample
        pdf.loc[:, "st_outcome"] = st_outcome  # Variance for quality
        pdf.loc[:, "stability"] = pdf["st_alive"] * np.exp(-w * pdf["st_outcome"])
        return pdf

    def compute_quality(self, w_time, w_stability, dt_stability):
        cache_name = "{}/{}-SCORE-w_time{}-w_stability{}-dt_stability{}-NSample{}.pkl".format(self._cache_dir, self._tensor_name, w_time, w_stability, dt_stability, self._data.shape[0])
        if os.path.exists(cache_name):
            print("Found cache! Loading from {}".format(cache_name))
            self._pdata = pkl.load(open(cache_name, "rb"))
        else:
            pdf = self._data.copy(deep=True)
            # Clear the unwanted recordings
            pdf = pdf[pdf.src_motion.isin(self._motions) & pdf.dst_motion.isin(self._motions)]

            output = []
            for src_motion in self._motions:
                for dst_motion in self._motions:
                    if src_motion == dst_motion: continue
                    lookat = pdf[(pdf.src_motion == src_motion) &
                                 (pdf.dst_motion == dst_motion)].reset_index(drop=True)

                    # Consolidate the transition outcomes
                    lookat.loc[:, "outcomes"] = lookat["is_alive"] * (lookat["control_accuracy"] / lookat["effort"]) * np.exp(-w_time * lookat["duration"])
                    # Stability around each samples
                    lookat = self._compute_stability(lookat, w_stability, dt_stability)
                    output.append(lookat)
            output = pd.concat(output)

            output["quality"] = output["outcomes"] * output["stability"]
            self._pdata = output

            # save cache
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
            pkl.dump(self._pdata, open(cache_name, "wb"))
            print('Saved cache to', cache_name)

    def get_samples(self, src_motion, src_phase, dst_motion, dst_gphase, tolerance=0.005):
        '''
        NOTE! This function uses GLOBAL destination phase
        '''
        mask = (self.tmat_df.src_motion == src_motion) & (self.tmat_df.dst_motion == dst_motion) \
                & (self.tmat_df.src_phase.between(src_phase-tolerance, src_phase+tolerance)) \
                & (self.tmat_df.dst_gphase.between(dst_gphase-tolerance, dst_gphase+tolerance)) \

        return self.tmat_df[mask]

    def query(self, src_motion, src_phase, dst_motion, dt, alive_th=0.9, random=False):
        '''
        This function outputs the closest transition point given fixed source information and the time window dt (seconds)
        '''
        pdf = self._pdata[(self._pdata.src_motion == src_motion) &
                          (self._pdata.dst_motion == dst_motion)]

        dt = dt / self._cycle_duration[src_motion]

        pdf = pdf.loc[pdf.src_phase.between(src_phase, src_phase+dt)]
        pdf = pdf.loc[pdf.st_alive > alive_th]
        pdf = pdf.sort_values("quality", ascending=False).reset_index(drop=True)

        osrc_phase = None  # closest point to transition
        odst_gphase = None # target transition on the destination motion

        artifacts = {}
        # Transition exists
        if len(pdf.src_phase.values) > 0:
            idx = 0
            if random:
                topk = min(10, pdf.shape[0])
                idx = np.random.randint(0,topk)

            osrc_phase = pdf.src_phase.values[idx]
            odst_gphase = pdf.dst_gphase.values[idx]

            for col in ['quality', 'st_alive', 'during_pos', 'post_pos']:
                artifacts[col] = pdf[col].values[0]

        return osrc_phase, odst_gphase, artifacts

    def save_precomputed_tensor(self, fps, alive_th=0.9):
        iters = np.linspace(0, 1, fps+1)
        dt = iters[1] - iters[0]

        precomputed_transitions = {}
        for src_motion in self._motions:
            for dst_motion in self._motions:
                if src_motion == dst_motion: continue
                key = "{}_{}".format(src_motion, dst_motion)

                if key not in precomputed_transitions:
                    precomputed_transitions[key] = []

                osrc_phases = []
                odst_gphases = []
                quality_scores = []
                for src_phase in tqdm(iters):
                    osrc_phase, odst_gphase, artifacts = self.query(src_motion, src_phase, dst_motion, dt, alive_th=alive_th)
                    if osrc_phase and odst_gphase and not np.isnan(artifacts['quality']):
                        osrc_phases.append(osrc_phase)
                        odst_gphases.append(odst_gphase)
                        quality_scores.append(artifacts['quality'])

                if len(quality_scores) > 0:
                    quality_thres = np.median(quality_scores)
                    print(key, quality_thres, quality_scores)
                    for osrc_phase, odst_gphase, quality_score in zip(osrc_phases, odst_gphases, quality_scores):
                        if osrc_phase and odst_gphase and quality_score >= quality_thres:
                            precomputed_transitions[key].append((osrc_phase, odst_gphase, 0))

        pkl.dump(precomputed_transitions, open("{}.pkl".format(self._tensor_name), "wb"))
