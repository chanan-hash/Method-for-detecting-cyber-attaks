# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
#
# def create_time_windows(df, window_size=10):
#     """Create time windows from DataFrame"""
#     print("\nCreating time windows:")
#     print(f"Total packets: {len(df)}")
#     print(f"Time range: {df['Timestamp'].min():.2f} to {df['Timestamp'].max():.2f}")
#     print(f"Window size: {window_size} seconds")
#
#     df = df.sort_values('Timestamp')
#     start_time = df['Timestamp'].min()
#     end_time = df['Timestamp'].max()
#     windows = []
#
#     current_time = start_time
#     window_count = 0
#     while current_time < end_time:
#         window_end = current_time + window_size
#         window_df = df[(df['Timestamp'] >= current_time) &
#                       (df['Timestamp'] < window_end)]
#         if not window_df.empty:
#             window_count += 1
#             print(f"Window {window_count}: {len(window_df)} packets, "
#                   f"time range {current_time:.2f} to {window_end:.2f}")
#             windows.append(window_df)
#         current_time = window_end
#
#     print(f"Created {len(windows)} windows")
#     return windows
#
# def create_histogram(window_df, hist_size=1500):
#     """Create 2D histogram from window DataFrame"""
#     # Print window stats
#     print(f"\nCreating histogram for window with {len(window_df)} packets")
#     print(f"Time range: {window_df['Timestamp'].min():.2f} to {window_df['Timestamp'].max():.2f}")
#     print(f"Size range: {window_df['Size'].min()} to {window_df['Size'].max()}")
#
#     # Normalize timestamps within window
#     df = window_df.copy()
#     df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()
#     df['Timestamp'] = df['Timestamp'] / df['Timestamp'].max() * hist_size
#
#     # Filter packet sizes (keeping 1500 as max packet size)
#     df = df[df['Size'] <= 1500]
#     print(f"After size filtering: {len(df)} packets")
#
#     # Create 2D histogram with custom size
#     hist, _, _ = np.histogram2d(
#         df['Timestamp'],
#         df['Size'],
#         bins=hist_size,
#         range=[[0, hist_size], [0, 1500]]  # Note: y-axis still max 1500 for packet sizes
#     )
#
#     # Convert to binary
#     binary_hist = (hist > 0).astype(int)
#     print(f"Created histogram with {np.sum(binary_hist)} active cells")
#
#     return binary_hist
#
# def process_packet_data(input_df, window_size=10, hist_size=1500):
#     """Main function to process packet data"""
#     print("\nStarting packet data processing:")
#     windows = create_time_windows(input_df, window_size)
#     histograms = []
#
#     print("\nCreating histograms:")
#     for i, window_df in enumerate(windows, 1):
#         print(f"\nProcessing window {i} of {len(windows)}")
#         hist = create_histogram(window_df, hist_size)
#         histograms.append(hist)
#
#     print(f"\nCompleted processing: Created {len(histograms)} histograms")
#     return histograms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from typing import List

MTU = 1500  # Maximum Transmission Unit
DEFAULT_TPS = 60  # Default time per session in seconds


def create_time_windows(df, window_size=10):
    """Create time windows from DataFrame"""
    print("\nCreating time windows:")
    print(f"Total packets: {len(df)}")
    print(f"Time range: {df['Timestamp'].min():.2f} to {df['Timestamp'].max():.2f}")
    print(f"Window size: {window_size} seconds")

    df = df.sort_values('Timestamp')
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()
    windows = []

    current_time = start_time
    window_count = 0
    while current_time < end_time:
        window_end = current_time + window_size
        window_df = df[(df['Timestamp'] >= current_time) &
                       (df['Timestamp'] < window_end)]
        if not window_df.empty:
            window_count += 1
            print(f"Window {window_count}: {len(window_df)} packets, "
                  f"time range {current_time:.2f} to {window_end:.2f}")
            windows.append(window_df)
        current_time = window_end

    print(f"Created {len(windows)} windows")
    return windows


def create_histogram(window_df, hist_size=1500):
    """Create 2D histogram from window DataFrame"""
    # Print window stats
    print(f"\nCreating histogram for window with {len(window_df)} packets")
    print(f"Time range: {window_df['Timestamp'].min():.2f} to {window_df['Timestamp'].max():.2f}")
    print(f"Size range: {window_df['Size'].min()} to {window_df['Size'].max()}")

    # Normalize timestamps within window
    df = window_df.copy()
    df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()
    df['Timestamp'] = df['Timestamp'] / df['Timestamp'].max() * hist_size

    # Filter packet sizes (keeping 1500 as max packet size)
    df = df[df['Size'] <= 1500]
    print(f"After size filtering: {len(df)} packets")

    # Create 2D histogram with custom size
    hist, _, _ = np.histogram2d(
        df['Timestamp'],
        df['Size'],
        bins=hist_size,
        range=[[0, hist_size], [0, 1500]]  # Note: y-axis still max 1500 for packet sizes
    )

    # Convert to binary
    binary_hist = (hist > 0).astype(int)
    print(f"Created histogram with {np.sum(binary_hist)} active cells")

    return binary_hist


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


def process_directory(input_dir: str, output_dir: str, window_size: int, hist_size: int, show_plot: bool):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    for file in csv_files:
        file_path = os.path.join(input_dir, file)
        print(f"Reading traffic data from {file_path}")
        df = pd.read_csv(file_path)
        windows = create_time_windows(df, window_size)

        for i, window_df in enumerate(windows, 1):
            print(f"\nProcessing window {i} of {len(windows)}")
            hist = create_histogram(window_df, hist_size)
            output_path = os.path.join(output_dir, f'flowpic_{file}_{i}')
            save_flowpic(hist, output_path, show_plot)
            print(f"Saved FlowPic for window {i} of file {file}")

    print("Done!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate FlowPic images from network traffic CSV files in a directory')
    parser.add_argument('--input', type=str, required=True, help='Directory containing CSV files')
    parser.add_argument('--window_size', type=int, default=DEFAULT_TPS, help='Time per session in seconds')
    parser.add_argument('--hist_size', type=int, default=MTU, help='Histogram size')
    parser.add_argument('--show', action='store_true', help='Show the plot')

    args = parser.parse_args()

    output_dir = os.path.join(args.input, 'flowpics_output')

    process_directory(args.input, output_dir, args.window_size, args.hist_size, args.show)


if __name__ == '__main__':
    main()