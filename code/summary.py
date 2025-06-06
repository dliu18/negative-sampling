import os
import csv
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm 
from sys import argv

def get_last_metric_values_and_duration(event_file):
    event_accumulator = EventAccumulator(event_file)
    event_accumulator.Reload()

    # Get the tags (i.e., metric names) from the event file
    tags = event_accumulator.Tags()['scalars']

    # Extract the last recorded value and step for each metric
    last_values = {}
    final_steps = {}
    all_timestamps = []
    gpu_timestamps = []
    for tag in tags:
        scalar_events = event_accumulator.Scalars(tag)
        if scalar_events:
            last_values[tag] = scalar_events[-1].value
            if "metrics" or "GPU" in tag:
                metric_values = np.array([event.value for event in scalar_events])
                last_values[tag + " max"] = f"{np.max(metric_values):.4f}" 
                # last_values[tag + " max epoch"] = scalar_events[np.argmax(metric_values)].step
            final_steps[tag] = scalar_events[-1].step
            # print(scalar_events)
            # Collect timestamps for duration calculation
            all_timestamps.extend(event.wall_time for event in scalar_events)
            if "GPU" in tag:
                gpu_timestamps.extend(event.wall_time for event in scalar_events)
    
    # Calculate the duration based on timestamps
    if all_timestamps and gpu_timestamps:
        duration = float(max(gpu_timestamps) - min(all_timestamps))
        duration = f"{duration}"
    else:
        duration = 0

    # Extract the final step value for "metrics/MRR"
    final_step = final_steps.get("Loss/total_loss", None)
    

    # Debug time information
    # print(f"Start: {min(all_timestamps)}")
    # print(f"Finished training embeddings: {max(gpu_timestamps)}")
    # print(f"Finished script: {max(all_timestamps)}")
    # print(f"Duration: {duration}")
    # print("\n")

    return last_values, duration, final_step

def find_event_files(directory):
    event_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    return event_files

def create_csv_summary(log_directory, output_csv):
    event_files = find_event_files(log_directory)
    rows = []

    for event_file in tqdm(event_files):
        graph_name, base_model, loss_func, n_negative, lr, lam, n2v_p, n2v_q, time = event_file.split('/')[2:11]
        trial_info = {
            "Graph": graph_name,
            "Model": base_model,
            "Loss Function": loss_func,
            "n_negative": n_negative,
            "Learning Rate": lr,
            "lambda": lam,
            "n2v_p": n2v_p,
            "n2v_q": n2v_q,
            'time': time
        }
        last_values, duration, mrr_final_step = get_last_metric_values_and_duration(event_file)
        length = {'Duration': duration, 'Steps': mrr_final_step}
        trial_info.update(length)
        trial_info.update(last_values)
        rows.append(trial_info)

    # Write to CSV
    if rows:
        # Determine all metric names
        metric_names = []
        for row in rows:
            for key in row.keys():
                if key not in metric_names:
                    metric_names.append(key)


        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metric_names)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

if __name__ == '__main__':
    assert len(argv) >= 2
    dir_name = argv[1]
    print(dir_name)
    log_directory = 'runs/' +  dir_name # Replace with your log directory
    output_csv = f'summary-{dir_name}.csv'  # Replace with your desired output CSV file name
    create_csv_summary(log_directory, output_csv)
    