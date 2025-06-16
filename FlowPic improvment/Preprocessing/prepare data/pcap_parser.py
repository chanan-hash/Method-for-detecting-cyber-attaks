
# import os
# import csv
# import subprocess
# import time

# # # For this you need to install wire shark and add it to the path, and then check you have tshark in the path
# # # Usually it is in the path C:\Program Files\Wireshark\tshark.exe
# def find_tshark():
#     """Find tshark executable in common installation paths"""
#     common_paths = [
#         r"C:\Program Files\Wireshark\tshark.exe",
#         r"C:\Program Files (x86)\Wireshark\tshark.exe",
#         "/usr/bin/tshark",
#         "/usr/local/bin/tshark"
#     ]

#     for path in common_paths:
#         if os.path.exists(path):
#             return path

#     return "tshark"  # Default to hoping it's in PATH


# def process_pcap(pcap_path, tshark_path):
#     """Process a PCAP file using tshark for maximum speed"""
#     base_path = os.path.splitext(pcap_path)[0]
#     txt_path = f"{base_path}_features.txt"
#     csv_path = f"{base_path}_features.csv"

#     try:
#         # Construct tshark command to extract needed fields
#         cmd = [
#             tshark_path,
#             '-r', pcap_path,  # Read from file
#             '-T', 'fields',  # Output fields
#             '-E', 'separator=,',  # Use comma as separator
#             '-e', 'frame.time_epoch',
#             '-e', 'ip.src',
#             '-e', 'tcp.srcport',
#             '-e', 'udp.srcport',
#             '-e', 'ip.dst',
#             '-e', 'tcp.dstport',
#             '-e', 'udp.dstport',
#             '-e', 'ip.len',
#             '-e', 'ip.proto',
#             '-Y', 'ip'  # Only IP packets
#         ]

#         print(f"Processing: {pcap_path}")

#         # Run tshark
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             print(f"Error processing file: {result.stderr}")
#             return False

#         # Process the output
#         packets = []
#         lines = result.stdout.strip().split('\n')

#         for line in lines:
#             if not line.strip():
#                 continue

#             fields = line.strip().split(',')
#             if len(fields) < 8:
#                 continue

#             try:
#                 timestamp = float(fields[0])
#                 src_ip = fields[1]
#                 src_port = fields[2] or fields[3]  # Try TCP port first, then UDP
#                 dst_ip = fields[4]
#                 dst_port = fields[5] or fields[6]  # Try TCP port first, then UDP
#                 size = fields[7]
# #                size = int(fields[7]) if fields[7] else 0  # Gets length from IP header field

#                 # Convert timestamp to readable format
#                 # timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

#                 # Determine protocol (TCP/UDP)
#                 protocol = 'TCP' if fields[2] else 'UDP'


#                 packet_info = {
# #                    'Timestamp': timestamp_str,
#                     'Timestamp': timestamp, # Keep timestamp as float for sorting
#                     'SourceIP': src_ip,
#                     'SourcePort': src_port,
#                     'DestinationIP': dst_ip,
#                     'DestinationPort': dst_port,
#                     'Protocol': protocol,
#                     'Size': size,
#                 }
#                 packets.append(packet_info)

#             except (ValueError, IndexError) as e:
#                 continue

#         # Write to files if we have packets
#         if packets:
#             # Write CSV
#             with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#                 fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
#                               'DestinationPort', 'Protocol', 'Size']
#                 writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(packets)

#             # Write TXT
#             with open(txt_path, 'w', encoding='utf-8') as txt_file:
#                 txt_file.write(
#                     "Timestamp, SourceIP, SourcePort, DestinationIP, DestinationPort, Protocol, Size\n")
#                 for packet in packets:
#                     txt_file.write(f"{packet['Timestamp']}, {packet['SourceIP']}, {packet['SourcePort']}, "
#                                    f"{packet['DestinationIP']}, {packet['DestinationPort']}, {packet['Protocol']}, "
#                                    f"{packet['Size']}\n")

#             print(f"Processed {len(packets)} packets")
#             print(f"Output written to:\n{txt_path}\n{csv_path}")
#             return True

#         else:
#             print("No valid packets found in the PCAP file")
#             return False

#     except Exception as e:
#         print(f"Error processing PCAP file {pcap_path}: {str(e)}")
#         return False


# def process_directory(directory_path, tshark_path):
#     """Process all PCAP files in a directory"""
#     success_count = 0
#     failed_count = 0

#     # Get list of all PCAP files
#     pcap_files = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(('.pcap', '.pcapng')):
#                 pcap_files.append(os.path.join(root, file))

#     total_files = len(pcap_files)
#     print(f"\nFound {total_files} PCAP files to process")

#     # Process each file
#     for index, pcap_path in enumerate(pcap_files, 1):
#         print(f"\nProcessing file {index}/{total_files}: {pcap_path}")
#         if process_pcap(pcap_path, tshark_path):
#             success_count += 1
#         else:
#             failed_count += 1

#     print(f"\nProcessing complete:")
#     print(f"Successfully processed: {success_count} files")
#     print(f"Failed to process: {failed_count} files")
#     return success_count, failed_count

# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(description='Extract features from PCAP files')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Directory containing PCAP files or path to single PCAP file')
#     args = parser.parse_args()

#     # Find tshark
#     tshark_path = find_tshark()
#     print(f"Using tshark from: {tshark_path}")

#     start_time = time.time()
#     input_path = args.input

#     if os.path.isfile(input_path):
#         print(f"\nProcessing single file: {input_path}")
#         process_pcap(input_path, tshark_path)
#     elif os.path.isdir(input_path):
#         print(f"\nProcessing all PCAP files in directory: {input_path}")
#         success_count, failed_count = process_directory(input_path, tshark_path)
#     else:
#         print(f"Error: {input_path} is not a valid file or directory")

#     total_time = time.time() - start_time
#     print(f"\nTotal processing time: {total_time:.2f} seconds")

####### Using scapy code for fixing the video voip and von viop mixed types ######

import logging
import csv
import argparse
from pathlib import Path
from scapy.layers.inet import IP, TCP, UDP
from scapy.utils import PcapReader

logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract flow data from pcap files into CSV format.")
parser.add_argument('--input', required=True, help="Input directory containing pcap files.")
parser.add_argument('--output', required=True, help="Output directory for CSV files.")
args = parser.parse_args()

input_path = Path(args.input)
output_path = Path(args.output)
output_path.mkdir(parents=True, exist_ok=True)

# Define the fieldnames for the CSV file
fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
              'DestinationPort', 'Protocol', 'Size']

for i_pcap_file, pcap_file in enumerate(input_path.glob("*.pcap")):

    csv_file_path = output_path / f"{pcap_file.stem}.csv"

    # Skip files that have already been processed
    if not csv_file_path.exists():
        print(f"Processing file {i_pcap_file + 1}: {pcap_file}")

        with open(csv_file_path, "w", newline="") as csvfile:
            csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            csv_writer.writeheader()

            with PcapReader(str(pcap_file)) as pcap_reader:
                for packet in pcap_reader:
                    try:
                        # Extract fields from the packet
                        timestamp = packet.time

                        if packet.haslayer(IP):
                            source_ip = packet[IP].src
                            destination_ip = packet[IP].dst
                            size = packet[IP].len if hasattr(packet[IP], 'len') else 0
                        else:
                            continue

                        if packet.haslayer(TCP):
                            if getattr(packet[TCP], "load", 0) == 0:
                                continue
                            source_port = packet[TCP].sport
                            destination_port = packet[TCP].dport
                            protocol = "TCP"
                        elif packet.haslayer(UDP):
                            if packet.haslayer("DNS"):
                                continue
                            source_port = packet[UDP].sport
                            destination_port = packet[UDP].dport
                            protocol = "UDP"
                        else:
                            continue

                        # Write the extracted information to the CSV
                        csv_writer.writerow({
                            'Timestamp': timestamp,
                            'SourceIP': source_ip,
                            'SourcePort': source_port,
                            'DestinationIP': destination_ip,
                            'DestinationPort': destination_port,
                            'Protocol': protocol,
                            'Size': size
                        })
                    except Exception as e:
                        logger.error(f"Error processing packet: {e}")

        print(f"Finished processing {pcap_file}, saved to {csv_file_path}")


############# For machine learning model using more statistical feature ################
# This extraction on large pcap files (like file transfer) will take a lot of time hourse or days. 
# """
# Here we are going to extract more statistical features from the pcap files.
# This to try and use machine learning models to classify the traffic, using size and inter-arrival time features.
# This idea came by Tal Shapira, who is a researcher and made the flowpic paper.
# We are relying on the work from here:
# https://www.kaggle.com/datasets/guillaumefraysse/ucdavisquic
# """

# import logging
# import csv
# import argparse
# from pathlib import Path
# from collections import defaultdict
# from statistics import mean, median, stdev, variance
# from scapy.layers.inet import IP, TCP, UDP
# from scapy.utils import PcapReader

# logger = logging.getLogger(__name__)

# # Parse command-line arguments
# parser = argparse.ArgumentParser(description="Extract flow data from pcap files into CSV format.")
# parser.add_argument('--input', required=True, help="Input directory containing pcap files.")
# parser.add_argument('--output', required=True, help="Output directory for CSV files.")
# args = parser.parse_args()

# input_path = Path(args.input)
# output_path = Path(args.output)
# output_path.mkdir(parents=True, exist_ok=True)

# # Enhanced fieldnames with both packet-level and statistical features
# fieldnames = [
#     # Packet-level features
#     'Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP', 'DestinationPort', 
#     'Protocol', 'Size',
#     # Statistical features
#     'TotalPackets', 'TotalBytes', 'Duration',
#     'MinPacketSize', 'MaxPacketSize', 'MeanPacketSize', 'MedianPacketSize',
#     'StdPacketSize', 'PacketSizeVariance', 'PacketSizeSkew',
#     'MinInterArrival', 'MaxInterArrival', 'MeanInterArrival',
#     'MedianInterArrival', 'StdInterArrival', 'InterArrivalVariance',
#     'ByteRate', 'PacketRate',
#     'TcpPercentage', 'UdpPercentage',
#     'SmallPacketRatio', 'LargePacketRatio',
#     'NumberOfBursts', 'MeanBurstSize', 'MaxBurstSize',
#     'MeanBurstDuration', 'MaxBurstDuration'
# ]

# class PcapStats:
#     def __init__(self):
#         self.packets = []
#         self.first_timestamp = None
#         self.burst_threshold = 0.1  # 100ms threshold for burst detection
#         self.small_packet_threshold = 128  # bytes
#         self.large_packet_threshold = 1024  # bytes
#         self.current_burst = []
#         self.bursts = []
#         self.tcp_count = 0
#         self.udp_count = 0
        
#     def add_packet(self, timestamp, size, protocol):
#         if self.first_timestamp is None:
#             self.first_timestamp = timestamp
#             inter_arrival = 0
#         else:
#             inter_arrival = timestamp - self.last_timestamp
            
#             # Update burst detection
#             if inter_arrival <= self.burst_threshold:
#                 self.current_burst.append((timestamp, size))
#             else:
#                 if len(self.current_burst) > 1:
#                     self.bursts.append(self.current_burst)
#                 self.current_burst = [(timestamp, size)]
        
#         self.last_timestamp = timestamp
#         self.packets.append((timestamp, size, inter_arrival))
        
#         # Update protocol counts
#         if protocol == "TCP":
#             self.tcp_count += 1
#         elif protocol == "UDP":
#             self.udp_count += 1

#     def _calculate_skewness(self, values):
#         if len(values) < 2:
#             return 0
#         mean_val = mean(values)
#         std_val = stdev(values)
#         if std_val == 0:
#             return 0
#         n = len(values)
#         return (sum((x - mean_val) ** 3 for x in values) / n) / (std_val ** 3)

#     def get_stats(self):
#         if not self.packets:
#             return dict.fromkeys([f for f in fieldnames if f not in 
#                                ['Timestamp', 'SourceIP', 'SourcePort', 
#                                 'DestinationIP', 'DestinationPort', 
#                                 'Protocol', 'Size']], 0)
        
#         try:
#             # Extract basic metrics
#             timestamps = [t for t, _, _ in self.packets]
#             sizes = [s for _, s, _ in self.packets]
#             inter_arrivals = [i for _, _, i in self.packets[1:]]
            
#             # Calculate duration
#             duration = max(timestamps) - min(timestamps)
#             duration = max(duration, 0.001)  # Avoid division by zero
            
#             # Finalize burst calculation
#             if self.current_burst and len(self.current_burst) > 1:
#                 self.bursts.append(self.current_burst)
            
#             # Calculate burst statistics
#             burst_sizes = [len(burst) for burst in self.bursts]
#             burst_durations = [(burst[-1][0] - burst[0][0]) for burst in self.bursts]
            
#             # Calculate total packets and protocol percentages
#             total_packets = len(self.packets)
#             tcp_percentage = (self.tcp_count / total_packets * 100) if total_packets > 0 else 0
#             udp_percentage = (self.udp_count / total_packets * 100) if total_packets > 0 else 0
            
#             return {
#                 # Statistical features
#                 'TotalPackets': total_packets,
#                 'TotalBytes': sum(sizes),
#                 'Duration': duration,
#                 'MinPacketSize': min(sizes),
#                 'MaxPacketSize': max(sizes),
#                 'MeanPacketSize': mean(sizes),
#                 'MedianPacketSize': median(sizes),
#                 'StdPacketSize': stdev(sizes) if len(sizes) > 1 else 0,
#                 'PacketSizeVariance': variance(sizes) if len(sizes) > 1 else 0,
#                 'PacketSizeSkew': self._calculate_skewness(sizes),
#                 'MinInterArrival': min(inter_arrivals) if inter_arrivals else 0,
#                 'MaxInterArrival': max(inter_arrivals) if inter_arrivals else 0,
#                 'MeanInterArrival': mean(inter_arrivals) if inter_arrivals else 0,
#                 'MedianInterArrival': median(inter_arrivals) if inter_arrivals else 0,
#                 'StdInterArrival': stdev(inter_arrivals) if len(inter_arrivals) > 1 else 0,
#                 'InterArrivalVariance': variance(inter_arrivals) if len(inter_arrivals) > 1 else 0,
#                 'ByteRate': sum(sizes) / duration,
#                 'PacketRate': len(sizes) / duration,
#                 'TcpPercentage': tcp_percentage,
#                 'UdpPercentage': udp_percentage,
#                 'SmallPacketRatio': sum(1 for s in sizes if s < self.small_packet_threshold) / len(sizes),
#                 'LargePacketRatio': sum(1 for s in sizes if s > self.large_packet_threshold) / len(sizes),
#                 'NumberOfBursts': len(self.bursts),
#                 'MeanBurstSize': mean(burst_sizes) if burst_sizes else 0,
#                 'MaxBurstSize': max(burst_sizes) if burst_sizes else 0,
#                 'MeanBurstDuration': mean(burst_durations) if burst_durations else 0,
#                 'MaxBurstDuration': max(burst_durations) if burst_durations else 0
#             }
#         except Exception as e:
#             logger.error(f"Error calculating statistics: {e}")
#             return dict.fromkeys([f for f in fieldnames if f not in 
#                                ['Timestamp', 'SourceIP', 'SourcePort', 
#                                 'DestinationIP', 'DestinationPort', 
#                                 'Protocol', 'Size']], 0)

# for i_pcap_file, pcap_file in enumerate(input_path.glob("*.pcap")):
#     csv_file_path = output_path / f"{pcap_file.stem}.csv"

#     # Skip files that have already been processed
#     if not csv_file_path.exists():
#         print(f"Processing file {i_pcap_file + 1}: {pcap_file}")

#         pcap_stats = PcapStats()
        
#         with open(csv_file_path, "w", newline="") as csvfile:
#             csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             csv_writer.writeheader()

#             with PcapReader(str(pcap_file)) as pcap_reader:
#                 for packet in pcap_reader:
#                     try:
#                         # Extract basic fields from the packet
#                         timestamp = packet.time

#                         if packet.haslayer(IP):
#                             source_ip = packet[IP].src
#                             destination_ip = packet[IP].dst
#                             size = packet[IP].len if hasattr(packet[IP], 'len') else 0
#                         else:
#                             continue

#                         if packet.haslayer(TCP):
#                             if getattr(packet[TCP], "load", 0) == 0:
#                                 continue
#                             source_port = packet[TCP].sport
#                             destination_port = packet[TCP].dport
#                             protocol = "TCP"
#                         elif packet.haslayer(UDP):
#                             if packet.haslayer("DNS"):
#                                 continue
#                             source_port = packet[UDP].sport
#                             destination_port = packet[UDP].dport
#                             protocol = "UDP"
#                         else:
#                             continue

#                         # Update statistics
#                         pcap_stats.add_packet(timestamp, size, protocol)
                        
#                         # Get current statistics
#                         stats = pcap_stats.get_stats()

#                         # Write row with both packet-level and statistical features
#                         csv_writer.writerow({
#                             # Packet-level features
#                             'Timestamp': timestamp,
#                             'SourceIP': source_ip,
#                             'SourcePort': source_port,
#                             'DestinationIP': destination_ip,
#                             'DestinationPort': destination_port,
#                             'Protocol': protocol,
#                             'Size': size,
#                             # Statistical features
#                             **stats
#                         })
#                     except Exception as e:
#                         logger.error(f"Error processing packet: {e}")

#         print(f"Finished processing {pcap_file}, saved to {csv_file_path}")