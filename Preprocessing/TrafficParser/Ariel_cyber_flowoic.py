import os
import sys

# Add multiple possible Npcap directories to PATH (Windows fix)
if sys.platform == "win32":
    possible_paths = [
        r'C:\Program Files\Wireshark',
        r'C:\Program Files\Npcap',
        r'C:\Windows\System32\Npcap',
        r'C:\Program Files\Wireshark\Npcap'
    ]

    # Add all existing paths to PATH
    for path in possible_paths:
        if os.path.exists(path):
            os.environ['PATH'] = path + os.pathsep + os.environ['PATH']
            print(f"Added Npcap path: {path}")

from nfstream import NFPlugin, NFStreamer
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


class FlowPic(NFPlugin):
    '''FlowPic | An NxN histogram of packet sizes and relative times, represented as an image.'''

    def __init__(self, save_path, time_per_subflow=30, min_time_per_subflow=None,
                 min_packets_per_subflow=10, image_dims=(1500, 1500), MTU=1500):
        self.save_path = save_path
        self.time_per_subflow = time_per_subflow
        self.min_time_per_subflow = int(np.ceil(
            time_per_subflow - time_per_subflow * 0.1666667)) if min_time_per_subflow == None else min_time_per_subflow
        self.min_packets_per_subflow = min_packets_per_subflow
        self.image_dims = image_dims
        self.mtu = MTU

    def on_init(self, packet, flow):
        flow.udps.src2dst_flowpic_data = _FlowPicInnerPlugin(flow.src2dst_first_seen_ms, self.mtu)
        flow.udps.dst2src_flowpic_data = None
        self.on_update(packet, flow)

    def on_update(self, packet, flow):
        if packet.ip_size == 0 or packet.ip_size > self.image_dims[0]:
            return

        if packet.direction == 0:  # src -> dst
            if self.is_subflow_ended(flow.udps.src2dst_flowpic_data.start_time, packet.time, self.time_per_subflow):
                if flow.udps.src2dst_flowpic_data.count > self.min_packets_per_subflow:
                    self.save_histogram(flow.udps.src2dst_flowpic_data.on_expire(),
                                        flow.udps.src2dst_flowpic_data.start_time,
                                        packet.direction,
                                        flow)
                flow.udps.src2dst_flowpic_data = _FlowPicInnerPlugin(packet.time, self.mtu)
            flow.udps.src2dst_flowpic_data.on_update(packet)
        else:
            if flow.udps.dst2src_flowpic_data == None:
                flow.udps.dst2src_flowpic_data = _FlowPicInnerPlugin(flow.dst2src_first_seen_ms, self.mtu)
            if self.is_subflow_ended(flow.udps.dst2src_flowpic_data.start_time, packet.time, self.time_per_subflow):
                if flow.udps.dst2src_flowpic_data.count > self.min_packets_per_subflow:
                    self.save_histogram(flow.udps.dst2src_flowpic_data.on_expire(),
                                        flow.udps.dst2src_flowpic_data.start_time,
                                        packet.direction,
                                        flow)
                flow.udps.dst2src_flowpic_data = _FlowPicInnerPlugin(packet.time, self.mtu)
            flow.udps.dst2src_flowpic_data.on_update(packet)

    def on_expire(self, flow):
        # SRC -> DST
        if (flow.src2dst_packets > 0 and
                flow.udps.src2dst_flowpic_data.count > self.min_packets_per_subflow and
                self.is_subflow_ended(flow.udps.src2dst_flowpic_data.start_time,
                                      flow.src2dst_last_seen_ms, self.min_time_per_subflow)):
            self.save_histogram(flow.udps.src2dst_flowpic_data.on_expire(),
                                flow.udps.src2dst_flowpic_data.start_time,
                                0,
                                flow)
        # DST -> SRC
        if (flow.dst2src_packets > 0 and
                flow.udps.dst2src_flowpic_data.count > self.min_packets_per_subflow and
                self.is_subflow_ended(flow.udps.dst2src_flowpic_data.start_time,
                                      flow.dst2src_last_seen_ms, self.min_time_per_subflow)):
            self.save_histogram(flow.udps.dst2src_flowpic_data.on_expire(),
                                flow.udps.dst2src_flowpic_data.start_time,
                                1,
                                flow)

        # cleanup
        del flow.udps.src2dst_flowpic_data
        del flow.udps.dst2src_flowpic_data

    def save_histogram(self, hist, subflow_start_time, direction, flow):
        filename = '-'.join(['flowpic',
                             str(subflow_start_time),
                             flow.src_ip,
                             str(flow.src_port),
                             flow.dst_ip,
                             str(flow.dst_port),
                             str(flow.protocol),
                             'src2dst' if direction == 0 else 'dst2src'])
        np.savez_compressed(os.path.join(self.save_path, filename), flowpic=hist)

    def is_subflow_ended(self, start_time_ms, current_time_ms, time_per_subflow_s):
        return (current_time_ms - start_time_ms) / 1000 > time_per_subflow_s


class _FlowPicInnerPlugin:
    def __init__(self, subflow_start_time, mtu) -> None:
        self.start_time = subflow_start_time
        self.sizes = []
        self.timestamps = []
        self.mtu = mtu
        self.count = 0

    def on_update(self, packet):
        self.count += 1
        relative_time_ms = packet.time - self.start_time
        relative_time_s = relative_time_ms / 1000
        self.sizes.append(packet.ip_size)
        self.timestamps.append(relative_time_s)

    def on_expire(self):
        return self._flow_2d_histogram(self.sizes, self.timestamps)

    def _flow_2d_histogram(self, sizes, ts):
        ts_norm = ((np.array(ts) - ts[0]) / (ts[-1] - ts[0])) * self.mtu
        H, _, _ = np.histogram2d(sizes, ts_norm,
                                 bins=(range(0, self.mtu + 1, 1),
                                       range(0, self.mtu + 1, 1)))
        return H.astype(np.uint16)


def process_input(input_source, output_dir="flowpic_outputs", csv_path="flow_features.csv"):
    """Process a PCAP file or network interface and generate FlowPic images and flow features CSV."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Convert path to absolute path and handle Unicode characters
        if os.path.exists(input_source):
            input_source = os.path.abspath(input_source)
            print(f"Processing file: {input_source}")

        # Initialize NFStreamer with FlowPic plugin
        streamer = NFStreamer(
            source=str(input_source),
            snapshot_length=65535,  # Ensure full packet capture
            decode_tunnels=True,  # Handle encapsulated packets
            plugins=[FlowPic(
                save_path=str(output_path),
                time_per_subflow=30,
                image_dims=(1500, 1500)
            )]
        )

        # Process flows and collect features
        flow_count = 0
        flow_features = []

        for flow in streamer:
            flow_count += 1
            # Extract relevant features from the flow
            features = {
                'src_ip': flow.src_ip,
                'dst_ip': flow.dst_ip,
                'src_port': flow.src_port,
                'dst_port': flow.dst_port,
                'protocol': flow.protocol,
                'duration_ms': flow.bidirectional_duration_ms,
                'packets': flow.bidirectional_packets,
                'bytes': flow.bidirectional_bytes,
                'min_ps': flow.bidirectional_min_ps,
                'max_ps': flow.bidirectional_max_ps,
                'mean_ps': flow.bidirectional_mean_ps,
                'min_piat_ms': flow.bidirectional_min_piat_ms,
                'max_piat_ms': flow.bidirectional_max_piat_ms,
                'mean_piat_ms': flow.bidirectional_mean_piat_ms,
                'flowpic_count': len(list(output_path.glob(
                    f'flowpic-*-{flow.src_ip}-{flow.src_port}-{flow.dst_ip}-{flow.dst_port}-{flow.protocol}-*.npz')))
            }
            flow_features.append(features)

        # Save features to CSV
        import pandas as pd
        df = pd.DataFrame(flow_features)
        df.to_csv(csv_path, index=False)
        print(f"Flow features saved to: {csv_path}")
        print(f"Processed {flow_count} flows")

        return output_path

    except Exception as e:
        print(f"Error processing PCAP: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Try copying the PCAP file to a path without special characters")
        print("2. Verify the PCAP file is not corrupted using Wireshark")
        print("3. Check if the PCAP file is readable using:")
        print("   tcpdump -r your_file.pcap")
        print("4. Try converting the PCAP using editcap:")
        print("   editcap -F pcap old.pcap new.pcap")
        raise

    print(f"Processed {flow_count} flows")
    return output_path


def view_flowpic(npz_file):
    """Load and display a FlowPic from an NPZ file."""
    flowpic = np.load(npz_file)['flowpic']
    plt.figure(figsize=(10, 10))
    plt.imshow(flowpic, cmap='viridis')
    plt.colorbar()
    plt.title('FlowPic Visualization')
    plt.xlabel('Normalized Time')
    plt.ylabel('Packet Size')
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate FlowPic images and extract features from network traffic')
    parser.add_argument('--input', type=str, required=True,
                        help='Input source (PCAP file path or network interface name like "eth0")')
    parser.add_argument('--output', type=str, default='flowpic_outputs',
                        help='Output directory for FlowPic images (default: flowpic_outputs)')
    parser.add_argument('--csv', type=str, default='flow_features.csv',
                        help='Output CSV file for flow features (default: flow_features.csv)')
    parser.add_argument('--view', action='store_true',
                        help='View the first generated FlowPic image after processing')

    args = parser.parse_args()

    # Process the input source
    output_dir = process_input(args.input, args.output, args.csv)
    print(f"FlowPic images saved to: {output_dir}")

    # View first FlowPic if requested
    if args.view:
        npz_files = list(output_dir.glob('*.npz'))
        if npz_files:
            print(f"Displaying first FlowPic from: {npz_files[0]}")
            view_flowpic(npz_files[0])
        else:
            print("No FlowPic images were generated")