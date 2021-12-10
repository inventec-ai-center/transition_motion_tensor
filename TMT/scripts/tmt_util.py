import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import os
import pickle as pkl
import re
import json
from tqdm import tqdm


def get_bin(x, bins):
    return np.argmin(np.abs(bins - x))


def bin2int(x):
    N = len(x)
    y = 0
    for i in range(N):
        p = N-1-i
        y += int(x[i]) * 2 ** p
    return y


def load_json(filepath):
    try:
        f = json.load(open(filepath, "r"))
    except Exception as e:
        print("Unable to read the {}".format(filepath))
        print(e)
        f = None
    return f


def map_phase(x, N, cycle_markers):
    cycle_markers = np.array(cycle_markers)
    pos = (x * N)
    dist = pos - cycle_markers

    # Compute cycle length
    cycle_length = np.mean(cycle_markers[1:] - cycle_markers[:-1])

    # Get last positive index
    idx = np.where(dist>0)[0]

    if len(idx) == 0:
        return (cycle_length - (cycle_markers[0] - pos)) / cycle_length

    elif len(idx) == len(cycle_markers):
        return (pos - cycle_markers[-1]) / cycle_length

    else:
        # Get last positive index
        idx = idx[-1]

        # Compute local phase
        local_phase = dist[idx] / (cycle_markers[idx+1] - cycle_markers[idx])
        return local_phase


def local2global(local_phase, action):
    map_phase_memo = {
        'Pace':   {'N': 5020,'cycle_markers': [160, 540, 930, 1320, 1700, 2090, 2480, 2860, 3250, 3640, 4020, 4410, 4800]},
        'Trot':   {'N': 4940,'cycle_markers': [270, 560, 850, 1140, 1430, 1720, 2010, 2300, 2590, 2881, 3173, 3465, 3757, 4049, 4340, 4630, 4920]},
        'Canter': {'N': 4980,'cycle_markers': [74, 329, 595, 855, 1115, 1385, 1645, 1905, 2166, 2434, 2695, 2955, 3224, 3485, 3745, 4014, 4275, 4535, 4805]}
    }

    if action not in map_phase_memo:
        return local_phase

    N = map_phase_memo[action]["N"]
    cycle_markers = np.array(map_phase_memo[action]["cycle_markers"])

    # Find one of the reference cycle
    N_markers = len(cycle_markers)
    idx = np.random.choice(range(N_markers-1), 1, replace=False)
    start = cycle_markers[idx]
    end = cycle_markers[idx+1]

    # Compute the global phase using the selected reference
    global_phase = ((end - start) * local_phase) / N
    return float(global_phase)


def global2local(global_phase, action):
    map_phase_memo = {
        'Pace':   {'N': 5020,'cycle_markers': [160, 540, 930, 1320, 1700, 2090, 2480, 2860, 3250, 3640, 4020, 4410, 4800]},
        'Trot':   {'N': 4940,'cycle_markers': [270, 560, 850, 1140, 1430, 1720, 2010, 2300, 2590, 2881, 3173, 3465, 3757, 4049, 4340, 4630, 4920]},
        'Canter': {'N': 4980,'cycle_markers': [74, 329, 595, 855, 1115, 1385, 1645, 1905, 2166, 2434, 2695, 2955, 3224, 3485, 3745, 4014, 4275, 4535, 4805]}
    }

    if action not in map_phase_memo:
        return global_phase

    return map_phase(global_phase, map_phase_memo[action]["N"], map_phase_memo[action]["cycle_markers"])


def find_cycles_from_contact(foot_contacts, heights, control_rewards, motion):
    gait_pattern = {
#         Dog
#         "Pace":   "f[hop](.){1,5}kkkk+(.){1,5}[abcd]f{1,5}",
#         "Trot":   "a.j+.aa.g+",
#         "Canter": "b[c,d][a-h]+i(.)+?e",
#         "Jump":   "ae+([h-z])+?[abcd]([a-h])+?([h-z])+?([a-h])+?a",
        "Pace":'(5(-3)*(-7)+(-\d+){1,2}(-10){2}(-\d+){1,2})',
        "Trot": "0(-\d){2}-9(.)*?-[01](.)*?6(.)*?0",
        "Canter": '(1-[2,3](-[0-7])+-8(-\d+)*?-4)',
        "Jump": '(0(-4)+(.){4,10}?[789](.){4,10}?[0123](.){4,10}?[789](.){4,10}?[23](.){4,10}?0)',
#         Humanoid
        "Walk":     "ac+[abcd]{1,2}b+",
        "Jog":      "ab+[abcd]{1,2}b+",
        "Run":      "ac{3,10}a+b+",
        "Backflip": "a[bc]{0,2}d{5,100}[bc]{0,2}a",
    }
    
    cycles = []
    if motion in ['Pace', 'Trot', 'Canter', 'Jump']:
        reg_string = "-".join(list(foot_contacts.astype("str")))
        reg_pattern = gait_pattern[motion]

        for i in re.finditer(reg_pattern, reg_string):
            idx = i.span()[0]
            idx = reg_string[:idx].count("-")
            cycles.append(idx)
        cycles = np.array(cycles)

        if len(cycles) == 0: return cycles

        # Prune based on the cycle-to-cycle distance
        dist = cycles[1:] - cycles[:-1]
        height = foot_contacts[cycles]
        
        # Post-process
        if motion == "Jump":
            p = []
            for i in range(0, len(cycles)-1):
                if cycles[i] < 4: continue
                d = cycles[i+1] - cycles[i]
                p.append(cycles[i])
            p.append(cycles[-1])
            cycles = np.array(p)
        elif motion == "Trot":
            p = []
            for i in range(0, len(cycles)-1):
                d = cycles[i+1] - cycles[i]
                th = (np.median(dist)*1.5)
                if height[i] != 0: continue

                p.append(cycles[i])
            p.append(cycles[-1])
            cycles = np.array(p)
        elif motion in ["Canter"]:
            p = []
            for i in range(0, len(cycles)-1):
                d = cycles[i+1] - cycles[i]
                th = (np.median(dist)*1.5)
                th_height = np.median(height)*1.5
                if d > th: continue
                if height[i] > th_height: continue

                p.append(cycles[i])
            p.append(cycles[-1])
            cycles = np.array(p)
    else :
        # Convert foot contacts for the string matching
        foot_contacts = "".join([chr(x + ord('a')) for x in foot_contacts])

        # Process the string
        if motion in gait_pattern.keys():
            for i in re.finditer(gait_pattern[motion], foot_contacts):
                # Get the start index for each match
                cycles.append(i.span()[0])
    return np.array(cycles)


# def find_cycles_from_contact(foot_contacts, heights, control_rewards, motion):
#     # From: pattern_matching_for_template_motion.ipynb
#     patterns = {
#         "Pace":'(5(-3)*(-7)+(-\d+){1,2}(-10){2}(-\d+){1,2})',
#         "Trot": "0(-\d){2}-9(.)*?-[01](.)*?6(.)*?0",
#         "Canter": '(1-[2,3](-[0-7])+-8(-\d+)*?-4)',
#         "Jump": '(0(-4)+(.){4,10}?[789](.){4,10}?[0123](.){4,10}?[789](.){4,10}?[23](.){4,10}?0)',
#     }

#     pos = []
#     if motion in patterns:
#         reg_string = "-".join(list(foot_contacts.astype("str")))
#         reg_pattern = patterns[motion]

#         for i in re.finditer(reg_pattern, reg_string):
#             idx = i.span()[0]
#             idx = reg_string[:idx].count("-")
#             pos.append(idx)
#         pos = np.array(pos)

#         if len(pos) == 0: return pos
#         # Prune based on the cycle-to-cycle distance
#         dist = pos[1:] - pos[:-1]
#         height = foot_contacts[pos]

#         if motion == "Jump":
#             # Post-process
#             p = []
#             for i in range(0, len(pos)-1):
#                 if pos[i] < 4: continue
#                 d = pos[i+1] - pos[i]
#                 p.append(pos[i])
#             p.append(pos[-1])
#             pos = np.array(p)

#         elif motion == "Trot":
#             # Post-process
#             p = []
#             for i in range(0, len(pos)-1):
#                 d = pos[i+1] - pos[i]
#                 th = (np.median(dist)*1.5)
#     #             if d > th: continue
#                 if height[i] != 0: continue

#                 p.append(pos[i])
#             p.append(pos[-1])
#             pos = np.array(p)

#         elif motion in ["Canter"]:
#             # Post-process
#             p = []
#             for i in range(0, len(pos)-1):
#                 d = pos[i+1] - pos[i]
#                 th = (np.median(dist)*1.5)
#                 th_height = np.median(height)*1.5
#                 if d > th: continue
#                 if height[i] > th_height: continue

#                 p.append(pos[i])
#             p.append(pos[-1])
#             pos = np.array(p)
#     else:

#         if motion == 'Stand':
#             _pos = []
#             for i in range(len(foot_contacts)):
#                 if (foot_contacts[i] == 15) and (np.abs(heights[i] - 0.3) < 0.02):
#                     _pos.append(i)

#             look_ahead = 3
#             pos = []
#             for i in range(0, len(_pos)-look_ahead):
#                 flag = True
#                 for j in range(look_ahead):
#                     if _pos[i + j] + 1 != _pos[i + j + 1]:
#                         flag = False
#                         break
#                 if flag:
#                     pos.append(_pos[i])

#         elif motion == 'Sit':
#             _pos = []
#             for i in range(len(foot_contacts)):
#                 if (foot_contacts[i] >= 12) and heights[i] < 0.2:
#                     _pos.append(i)

#             pos = []
#             for i, p in enumerate(_pos):
#                 if p + 1 < len(control_rewards) and np.abs(control_rewards[p] - control_rewards[p+1]) < 5e-3:
#                     pos.append(p)
                    
#         elif motion == 'SitStand':
#             pos = [0]

#     return np.array(pos)
