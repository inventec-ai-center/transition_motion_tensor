import os
import argparse
import time
from glob import glob
import sys
sys.path.append("./scripts/")
from transition_motion_tensor import TransitionMotionTensor as TMT


def main(args):
    w_stability = float(args['w_stability'])
    w_time = float(args['w_time'])
    dt_stability = float(args['dt_stability'])
    alive_thres = float(args['alive_thres'])
    input_path = str(args['input_path'])
    tensor_name = os.path.splitext(os.path.basename(input_path))[0]

    transition_files = glob("{0}/*.json".format(input_path))
    tensor = TMT(tensor_name, cache_dir='./data/cache')
    tensor.load_from_json(transition_files)
    tensor.compute_quality(w_time=w_time, w_stability=w_stability, dt_stability=dt_stability)
    tensor.save_precomputed_tensor(fps=30, alive_th=alive_thres)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w_stability', default=0.015)
    parser.add_argument('--w_time', default=1)
    parser.add_argument('--dt_stability', default=0.01)
    parser.add_argument('--alive_thres', default=0.9)
    parser.add_argument('--input_path', required=True)
    args = parser.parse_args()

    start = time.time()
    main(vars(args))
    elapsed_time = time.time() - start
    print('Finished in %.1f seconds.' % (elapsed_time))
