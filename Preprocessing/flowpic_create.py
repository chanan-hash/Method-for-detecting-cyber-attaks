"""
flowpic_generator.py creates FlowPic images from network traffic CSV files.
Saves both PNG visualizations and NPY arrays for further processing.
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

MTU = 1500  # Maximum Transmission Unit
DEFAULT_TPS = 60  # Default time per session in seconds

def read_traffic_csv(file_path: str) -> Tuple[List[float], List[int]]:
    timestamps = []
    sizes = []

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)

        for row in reader:
            try:
                timestamps.append(float(row[0]))
                size = int(row[6])
                direction = row[7].lower()

                if direction in ["out", "outgoing"]:
                    size = -size
                sizes.append(size)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed row: {row}")
                continue

    return np.array(timestamps), np.array(sizes)

def create_flowpic(timestamps: np.ndarray, sizes: np.ndarray, tps: int = DEFAULT_TPS) -> np.ndarray:
    max_delta_time = tps if tps else timestamps[-1] - timestamps[0]
    ts_norm = ((timestamps - timestamps[0]) / max_delta_time) * MTU

    bins_x = np.linspace(-MTU, MTU, MTU)
    bins_y = np.linspace(0, MTU, MTU)
    H, xedges, yedges = np.histogram2d(
        sizes,
        ts_norm,
        bins=[bins_x, bins_y]
    )

    return H.astype(np.uint16)

def save_flowpic(flowpic: np.ndarray, output_path: str, show_plot: bool = False):
    np.save(output_path + '.npy', flowpic)

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(flowpic, cmap='binary')
    plt.colorbar()
    plt.xlabel('Normalized Time')
    plt.ylabel('Packet Size')
    plt.title('FlowPic Visualization')

    plt.gca().set_aspect('equal')
    plt.xlim(0, flowpic.shape[1])
    plt.ylim(0, flowpic.shape[0])

    plt.tight_layout()
    plt.savefig(output_path + '.png', bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate FlowPic images from network traffic CSV')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output', type=str, help='Output directory path')
    parser.add_argument('--tps', type=int, default=DEFAULT_TPS, help='Time per session in seconds')
    parser.add_argument('--show', action='store_true', help='Show the plot')

    args = parser.parse_args()

    if not args.output:
        args.output = os.path.splitext(args.input)[0] + '_flowpics'

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    print(f"Reading traffic data from {args.input}")
    timestamps, sizes = read_traffic_csv(args.input)

    start_time = timestamps[0]
    end_time = timestamps[-1]
    current_time = start_time

    while current_time < end_time:
        interval_mask = (timestamps >= current_time) & (timestamps < current_time + args.tps)
        interval_timestamps = timestamps[interval_mask]
        interval_sizes = sizes[interval_mask]

        if len(interval_timestamps) > 0:
            flowpic = create_flowpic(interval_timestamps, interval_sizes, args.tps)
            output_path = os.path.join(args.output, f'flowpic_{int(current_time)}')
            save_flowpic(flowpic, output_path, args.show)
            print(f"Saved FlowPic for interval starting at {current_time}")

        current_time += args.tps

    print("Done!")

if __name__ == '__main__':
    main()
# python flowpic_generator.py --input "path/to/your/traffic.csv" --output "output_path" --tps 60 --show