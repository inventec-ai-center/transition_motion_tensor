import os
import argparse
import math
import subprocess
import time


def on_exp_finished_callback(input_files, output_files, stdout_files, output_path):
    args_path = output_path + '/args'

    if not os.path.exists(args_path):
        os.makedirs(args_path)

    # Move input_files to args_path
    for input_file in input_files:
        if os.path.isfile(input_file):
            os.rename(input_file, args_path + '/' + os.path.basename(input_file))
    return


def main(args):
    num_samples = int(args['num_samples'])
    num_workers = int(args['num_workers'])
    output_path = str(args['output_path'])
    stdout_path = output_path + '/stdout_logs'
    num_samples_per_workers = math.ceil(num_samples / num_workers)
    exp_filename = os.path.basename(output_path)
    template_args = ''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(stdout_path):
        os.makedirs(stdout_path)

    with open(str(args['input_filename']), 'r') as fin:
        template_args = fin.read()
    process_list = []
    output_files = []
    input_files  = []
    stdout_files = []
    for i in range(num_workers):
        output_filename = '%s/%s_worker%d.json' % (output_path, exp_filename, i)
        args_filename = './args/%s_worker%d.txt' % (exp_filename, i)
        rand_seed = int(time.time() + 1000 * i)
        args_str = template_args.replace('{NUM_TRANSITION_SAMPLES}', str(num_samples_per_workers)) \
                                .replace('{OUTPUT_FILENAME}', str(output_filename)) \
                                .replace('{RAND_SEED}', str(rand_seed))
        with open(args_filename, 'w') as fin:
            fin.write(args_str)

        proc = None
        args = ['python', 'TMT_Optimizer.py', '--arg_file', args_filename]
        if i == 0:
            proc = subprocess.Popen(args)
        else:
            stdout_filename = "%s/stdout_workers%d.txt" % (stdout_path, i)
            with open(stdout_filename, "wb") as out:
                proc = subprocess.Popen(args, stdout=out, stderr=out)
                stdout_files.append(stdout_filename)
        process_list.append(proc)
        input_files.append(args_filename)
        output_files.append(output_filename)

    while(True):
        num_proc_done = 0
        for i, proc in enumerate(process_list):
            if proc.poll() is not None:
                num_proc_done += 1

        if num_proc_done == len(process_list):
            on_exp_finished_callback(input_files, output_files, stdout_files, output_path)
            break

        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=1)
    parser.add_argument("--num_samples", required=True)
    parser.add_argument("--input_filename", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    start = time.time()
    main(vars(args))
    elapsed_time = time.time() - start
    print('Finished in %.1f seconds.' % (elapsed_time))
