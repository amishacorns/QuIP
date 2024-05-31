import subprocess
import multiprocessing as mp
import queue
import os
import time

AVAILABLE_GPUS = [0, 1, 2, 3, 4, 5]
RESULTS_DIR = 'results/'
PYTHON = '/home/jad443/.conda/envs/compressor/bin/python'
RUN_FILE = 'opt.py'

def worker(gpu_id, task_queue, model_dir, output_queue):
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return

        value, other_args = task
        output_file = os.path.join(model_dir, f'output_{value:.2f}_gpu{gpu_id}.txt')
        command = f"{PYTHON} {RUN_FILE} {other_args} --device cuda:{gpu_id}"
        print(command)
        with open(output_file, 'w') as file:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, _ = process.communicate()
            file.write(stdout.decode())
            output_queue.put(output_file)
     
p_values = [round(x * 0.05, 2) for x in range(21)]
print(p_values)
time_stamp = str(int(time.time()))

models = ['opt-125m']
time_dir = f"{RESULTS_DIR}/{time_stamp}"
os.makedirs(time_dir, exist_ok=True)
for model in models:
    model_dir = os.path.join(time_dir, model.replace('/', '_'))  # Replace '/' in model name with '_'
    os.makedirs(model_dir, exist_ok=True)
    task_queue = mp.Manager().Queue()
    output_queue = mp.Manager().Queue()

    for p in p_values:
        values = [-1, -p, p, 1]
        values = [val * 1.5 + 1.5 for val in values]
        values = [0, 1, 2, 3]
        values = ' '.join([str(val) for val in values])
        other_args = f"facebook/{model} wikitext2 --incoh_processing --wbits=2 --lookup_values {values}"
        # other_args = f"facebook/{model} wikitext2 --wbits=2 --incoh_processing"
        task_queue.put((p, other_args))

    processes = []
    for gpu_id in AVAILABLE_GPUS:
        p = mp.Process(target=worker, args=(gpu_id, task_queue, model_dir, output_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Collect and save results
    while not output_queue.empty():
        output_file = output_queue.get()
