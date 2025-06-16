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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def create_time_windows(df, window_size=60):
    """Create time windows from DataFrame"""
    df = df.sort_values('Timestamp')
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    windows = []
    current_time = start_time

    while current_time < end_time:
        window_end = current_time + window_size
        window_df = df[(df['Timestamp'] >= current_time) & (df['Timestamp'] < window_end)]
        if not window_df.empty:
            windows.append(window_df)
        current_time = window_end
 
    return windows

def create_time_windows(df, window_size=60, overlap=0):
    """Create time windows from DataFrame with overlap"""
    overlap_time = window_size * overlap
    df = df.sort_values('Timestamp')
    start_time = df['Timestamp'].min()
    end_time = df['Timestamp'].max()

    windows = []
    current_time = start_time

    while current_time < end_time:
        window_end = current_time + window_size
        window_df = df[(df['Timestamp'] >= current_time) & (df['Timestamp'] < window_end)]
        if not window_df.empty:
            windows.append(window_df)
        current_time = window_end - overlap_time
 
    return windows

def create_histogram(window_df, hist_size=1500):
    """Create 2D histogram from window DataFrame"""
    df = window_df.copy()

    # Normalize timestamps to range [0, hist_size]
    time_min = df['Timestamp'].min()
    time_range = df['Timestamp'].max() - time_min
    if time_range > 0:
        df['NormalizedTime'] = ((df['Timestamp'] - time_min) / time_range) * hist_size
    else:
        df['NormalizedTime'] = 0  # All timestamps are the same

    df = df[df['Size'] <= 1500]  # Filter packet sizes <= 1500

    hist, _, _ = np.histogram2d(
        df['NormalizedTime'],
        df['Size'],
        bins=hist_size,
        range=[[0, hist_size], [0, 1500]]
    )

    return (hist > 0).astype(int)  # Convert to binary

def process_packet_data(input_df, window_size=60, hist_size=1500, overlap=0):
    """Main function to process packet data"""
    print("\nStarting packet data processing:")
    windows = create_time_windows(input_df, window_size, overlap)
    histograms = []

    print("\nCreating histograms:")
    for i, window_df in enumerate(windows, 1):
        print(f"\nProcessing window {i} of {len(windows)}")
        hist = create_histogram(window_df, hist_size)
        histograms.append(hist)

    print(f"\nCompleted processing: Created {len(histograms)} histograms")
    return histograms

def save_histogram(hist, output_path, window_index):
    """Save histogram as PNG and NPY"""
    np.save(f"{output_path}_window_{window_index}.npy", hist)

    plt.figure(figsize=(10, 10))
    plt.pcolormesh(hist, cmap='binary', shading='auto')
    plt.colorbar()
    plt.xlabel('Normalized Time')
    plt.ylabel('Packet Size')
    plt.title(f'FlowPic - Window {window_index}')

    plt.gca().set_aspect('equal')
    plt.xlim(0, hist.shape[1])
    plt.ylim(0, hist.shape[0])

    plt.tight_layout()
    plt.savefig(f"{output_path}_window_{window_index}.png", bbox_inches='tight', dpi=300)
    plt.close()


def process_csv_to_flowpics(input_csv, output_dir, window_size=60, hist_size=1500):
    """Main function to process CSV and generate FlowPics"""
    df = pd.read_csv(input_csv)
    
    # df = pd.read_csv(input_csv, low_memory=False) # added low_memory=False

#    # checking for mixed types, help to debug
#     print(df.iloc[:, [2, 4]].head())  # Display first rows of columns 2 and 4
#     print(df.iloc[:, [2, 4]].dtypes)  # Check their inferred types

#     for col in df.columns:
#         if df[col].dtype == "object":
#             print(f"Column '{col}' has mixed types:")
#             print(df[col].unique()[:10])  # Show unique values

    # Ensure required columns are present
    if 'Timestamp' not in df.columns or 'Size' not in df.columns:
        raise ValueError("Input CSV must contain 'Timestamp' and 'Size' columns")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    windows = create_time_windows(df, window_size)

    for idx, window_df in enumerate(windows, 1):
        if not window_df.empty:
            hist = create_histogram(window_df, hist_size)
            save_histogram(hist, os.path.join(output_dir, "file_flowpic"), idx) # flowpic name can be changed according to the dataset type



# Adding direction dimention in the pictures
def create_histogram_with_direction(window_df, hist_size=1500):
    """
    Create a 2D histogram image incorporating the 'Direction' dimension.
    Packets with Direction==1 (client to server) will be light gray,
    and packets with Direction==0 (server to client) will be dark gray.
    If a bin contains packets from both directions, the light gray is applied.
    """
    # Copy the window for isolation
    df = window_df.copy()

    # Normalize timestamps to range [0, hist_size]
    time_min = df['Timestamp'].min()
    time_range = df['Timestamp'].max() - time_min
    if time_range > 0:
        df['NormalizedTime'] = ((df['Timestamp'] - time_min) / time_range) * hist_size
    else:
        df['NormalizedTime'] = 0  # All timestamps are the same

    # Filter packet sizes <=1500
    df = df[df['Size'] <= 1500]

    # Separate packets by direction
    df0 = df[df['Direction'] == 0]  # server->client (dark gray)
    df1 = df[df['Direction'] == 1]  # client->server (light gray)

    # Build 2D histograms for each direction
    hist0, xedges, yedges = np.histogram2d(
        df0['NormalizedTime'],
        df0['Size'],
        bins=hist_size,
        range=[[0, hist_size], [0, 1500]]
    )
    hist1, _, _ = np.histogram2d(
        df1['NormalizedTime'],
        df1['Size'],
        bins=hist_size,
        range=[[0, hist_size], [0, 1500]]
    )

    # Convert to binary (presence or absence of packets)
    bin0 = (hist0 > 0).astype(int)
    bin1 = (hist1 > 0).astype(int)

    # Create image: initialize to 0 (black)
    # We'll use pixel intensity 50 for dark gray (direction 0)
    # and 200 for light gray (direction 1)
    image = np.zeros(bin0.shape, dtype=np.uint8)

    # Set bins where direction 1 packets exist to light gray (200)
    image[bin1 == 1] = 200

    # For bins where there is only direction 0 (and not direction 1), set to dark gray (50)
    only0 = (bin0 == 1) & (bin1 == 0)
    image[only0] = 50

    return image

def save_histogram_with_direction(image, output_path, window_index):
    """
    Save the histogram image as PNG and NPY.
    """
    np.save(f"{output_path}_window_{window_index}.npy", image)

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray', origin='lower', aspect='equal')
    plt.colorbar()
    plt.xlabel('Normalized Time')
    plt.ylabel('Packet Size')
    plt.title(f'FlowPic with Direction - Window {window_index}')
    
    plt.tight_layout()
    plt.savefig(f"{output_path}_window_{window_index}.png", bbox_inches='tight', dpi=300)
    plt.close()


def process_csv_to_flowpics_with_direction(input_csv, output_dir, window_size=60, hist_size=1500):
    """
    Process CSV and generate FlowPics where the 'Direction' column is used
    to distinguish packets: 1→light gray, 0→dark gray.
    """
    df = pd.read_csv(input_csv)
    
    # Ensure required columns are present
    if 'Timestamp' not in df.columns or 'Size' not in df.columns or 'Direction' not in df.columns:
        raise ValueError("Input CSV must contain 'Timestamp', 'Size', and 'Direction' columns")

    # Create output directory if missing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    windows = create_time_windows(df, window_size)

    for idx, window_df in enumerate(windows, 1):
        if not window_df.empty:
            # Create a histogram image incorporating Direction
            image = create_histogram_with_direction(window_df, hist_size)
            save_histogram_with_direction(image, os.path.join(output_dir, "file_flowpic"), idx)

def process_packet_data_with_direction(input_df, window_size=60, hist_size=1500, overlap=0):
    """Main function to process packet data incorporating direction

    Creates time windows with possible overlap, then generates binary histograms
    using the packet 'Direction' to differentiate pixel intensities.
    """
    print("\nStarting packet data processing with direction:")
    windows = create_time_windows(input_df, window_size, overlap)
    histograms = []

    print("\nCreating histograms with direction:")
    for i, window_df in enumerate(windows, 1):
        print(f"\nProcessing window {i} of {len(windows)}")
        hist = create_histogram_with_direction(window_df, hist_size)
        histograms.append(hist)

    print(f"\nCompleted processing: Created {len(histograms)} histograms with direction")
    return histograms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV to FlowPics")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file path")
    parser.add_argument('--output', type=str, required=True, help="Output directory path")
    args = parser.parse_args()

    input_csv = args.input
    output_dir = args.output

    process_csv_to_flowpics(input_csv, output_dir)