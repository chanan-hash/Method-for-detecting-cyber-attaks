###################### scapy code - very slow ############################

# import os
# import csv
# from datetime import datetime
# from scapy.all import rdpcap, IP, TCP, UDP
# import time
#
#
# def determine_direction(src_ip, dst_ip):
#     """Determine packet direction (inbound/outbound) based on IP addresses"""
#     private_ip_prefixes = [
#         '10.',
#         '172.16.', '172.17.', '172.18.', '172.19.',
#         '172.20.', '172.21.', '172.22.', '172.23.',
#         '172.24.', '172.25.', '172.26.', '172.27.',
#         '172.28.', '172.29.', '172.30.', '172.31.',
#         '192.168.'
#     ]
#
#     src_is_private = any(src_ip.startswith(prefix) for prefix in private_ip_prefixes)
#     dst_is_private = any(dst_ip.startswith(prefix) for prefix in private_ip_prefixes)
#
#     if src_is_private and not dst_is_private:
#         return 'outbound'
#     elif not src_is_private and dst_is_private:
#         return 'inbound'
#     else:
#         return 'internal'
#
#
# def process_pcap(pcap_path):
#     """Process a PCAP file and extract features using scapy"""
#     base_path = os.path.splitext(pcap_path)[0]
#     txt_path = f"{base_path}_features.txt"
#     csv_path = f"{base_path}_features.csv"
#
#     packets = []
#     try:
#         # Use scapy to read the pcap file
#         print(f"Reading PCAP file: {pcap_path}")
#         pcap = rdpcap(pcap_path)
#         print(f"Successfully read {len(pcap)} packets")
#
#         for packet in pcap:
#             try:
#                 # Check if packet has IP layer
#                 if IP in packet:
#                     ip_layer = packet[IP]
#
#                     # Get protocol information
#                     if TCP in packet:
#                         proto_layer = packet[TCP]
#                         protocol = 'TCP'
#                     elif UDP in packet:
#                         proto_layer = packet[UDP]
#                         protocol = 'UDP'
#                     else:
#                         continue
#
#                     # Extract features
#                     timestamp = datetime.fromtimestamp(float(packet.time)).strftime('%Y-%m-%d %H:%M:%S.%f')
#                     src_ip = ip_layer.src
#                     dst_ip = ip_layer.dst
#                     src_port = proto_layer.sport
#                     dst_port = proto_layer.dport
#                     size = len(packet)
#                     direction = determine_direction(src_ip, dst_ip)
#
#                     packet_info = {
#                         'Timestamp': timestamp,
#                         'SourceIP': src_ip,
#                         'SourcePort': src_port,
#                         'DestinationIP': dst_ip,
#                         'DestinationPort': dst_port,
#                         'Protocol': protocol,
#                         'Size': size,
#                         'Direction': direction
#                     }
#                     packets.append(packet_info)
#
#             except Exception as e:
#                 print(f"Error processing individual packet: {e}")
#                 continue
#
#         if packets:
#             # Write to text file
#             with open(txt_path, 'w', encoding='utf-8') as txt_file:
#                 txt_file.write(
#                     "Timestamp, SourceIP, SourcePort, DestinationIP, DestinationPort, Protocol, Size, Direction\n")
#                 for packet in packets:
#                     txt_file.write(f"{packet['Timestamp']}, {packet['SourceIP']}, {packet['SourcePort']}, "
#                                    f"{packet['DestinationIP']}, {packet['DestinationPort']}, {packet['Protocol']}, "
#                                    f"{packet['Size']}, {packet['Direction']}\n")
#
#             # Write to CSV file
#             with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#                 fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
#                               'DestinationPort', 'Protocol', 'Size', 'Direction']
#                 writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(packets)
#
#             print(f"Successfully processed {len(packets)} packets")
#             print(f"Output written to:\n{txt_path}\n{csv_path}")
#         else:
#             print("No valid packets found in the PCAP file")
#
#     except Exception as e:
#         print(f"Error processing PCAP file {pcap_path}: {str(e)}")
#         return False
#
#     return True
#
#
# def process_directory(directory_path):
#     """Process all PCAP files in a directory"""
#     success_count = 0
#     failed_count = 0
#
#     # Get list of all PCAP files
#     pcap_files = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(('.pcap', '.pcapng')):
#                 pcap_files.append(os.path.join(root, file))
#
#     total_files = len(pcap_files)
#     print(f"\nFound {total_files} PCAP files to process")
#
#     # Process each file
#     for index, pcap_path in enumerate(pcap_files, 1):
#         print(f"\nProcessing file {index}/{total_files}: {pcap_path}")
#         if process_pcap(pcap_path):
#             success_count += 1
#         else:
#             failed_count += 1
#
#     print(f"\nProcessing complete:")
#     print(f"Successfully processed: {success_count} files")
#     print(f"Failed to process: {failed_count} files")
#     return success_count, failed_count
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Extract features from PCAP files')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Directory containing PCAP files or path to single PCAP file')
#     args = parser.parse_args()
#
#     start_time = time.time()
#     input_path = args.input
#
#     if os.path.isfile(input_path):
#         print(f"\nProcessing single file: {input_path}")
#         process_pcap(input_path)
#     elif os.path.isdir(input_path):
#         print(f"\nProcessing all PCAP files in directory: {input_path}")
#         success_count, failed_count = process_directory(input_path)
#     else:
#         print(f"Error: {input_path} is not a valid file or directory")
#
#     total_time = time.time() - start_time
#     print(f"\nTotal processing time: {total_time:.2f} seconds")


####################### Faster Scapy Code Using tshark ############################
import os
import csv
import subprocess
import time
import dpkt
from datetime import datetime

from numexpr.expressions import double


# For this you need to install wire shark and add it to the path, and then check you have tshark in the path
# Usually it is in the path C:\Program Files\Wireshark\tshark.exe
def find_tshark():
    """Find tshark executable in common installation paths"""
    common_paths = [
        r"C:\Program Files\Wireshark\tshark.exe",
        r"C:\Program Files (x86)\Wireshark\tshark.exe",
        "/usr/bin/tshark",
        "/usr/local/bin/tshark"
    ]

    for path in common_paths:
        if os.path.exists(path):
            return path

    return "tshark"  # Default to hoping it's in PATH


def process_pcap(pcap_path, tshark_path):
    """Process a PCAP file using tshark for maximum speed"""
    base_path = os.path.splitext(pcap_path)[0]
    txt_path = f"{base_path}_features.txt"
    csv_path = f"{base_path}_features.csv"

    try:
        # Construct tshark command to extract needed fields
        cmd = [
            tshark_path,
            '-r', pcap_path,  # Read from file
            '-T', 'fields',  # Output fields
            '-E', 'separator=,',  # Use comma as separator
            '-e', 'frame.time_epoch',
            '-e', 'ip.src',
            '-e', 'tcp.srcport',
            '-e', 'udp.srcport',
            '-e', 'ip.dst',
            '-e', 'tcp.dstport',
            '-e', 'udp.dstport',
            '-e', 'ip.len',
            '-e', 'ip.proto',
            '-Y', 'ip'  # Only IP packets
        ]

        print(f"Processing: {pcap_path}")

        # Run tshark
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error processing file: {result.stderr}")
            return False

        # Process the output
        packets = []
        lines = result.stdout.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            fields = line.strip().split(',')
            if len(fields) < 8:
                continue

            try:
                timestamp = float(fields[0])
                src_ip = fields[1]
                src_port = fields[2] or fields[3]  # Try TCP port first, then UDP
                dst_ip = fields[4]
                dst_port = fields[5] or fields[6]  # Try TCP port first, then UDP
                size = fields[7]
#                size = int(fields[7]) if fields[7] else 0  # Gets length from IP header field
                #size = double(fields[7]) if fields[7] else 0  # Gets length from IP header field

                # Convert timestamp to readable format
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')

                # Determine protocol (TCP/UDP)
                protocol = 'TCP' if fields[2] else 'UDP'

                # Determine direction
                direction = determine_direction(src_ip, dst_ip)

                packet_info = {
#                    'Timestamp': timestamp_str,
                    'Timestamp': timestamp, # Keep timestamp as float for sorting
                    'SourceIP': src_ip,
                    'SourcePort': src_port,
                    'DestinationIP': dst_ip,
                    'DestinationPort': dst_port,
                    'Protocol': protocol,
                    'Size': size,
                    'Direction': direction
                }
                packets.append(packet_info)

            except (ValueError, IndexError) as e:
                continue

        # Write to files if we have packets
        if packets:
            # Write CSV
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
                              'DestinationPort', 'Protocol', 'Size', 'Direction']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(packets)

            # Write TXT
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(
                    "Timestamp, SourceIP, SourcePort, DestinationIP, DestinationPort, Protocol, Size, Direction\n")
                for packet in packets:
                    txt_file.write(f"{packet['Timestamp']}, {packet['SourceIP']}, {packet['SourcePort']}, "
                                   f"{packet['DestinationIP']}, {packet['DestinationPort']}, {packet['Protocol']}, "
                                   f"{packet['Size']}, {packet['Direction']}\n")

            print(f"Processed {len(packets)} packets")
            print(f"Output written to:\n{txt_path}\n{csv_path}")
            return True

        else:
            print("No valid packets found in the PCAP file")
            return False

    except Exception as e:
        print(f"Error processing PCAP file {pcap_path}: {str(e)}")
        return False


def determine_direction(src_ip, dst_ip):
    """Determine packet direction (inbound/outbound) based on IP addresses"""
    private_ip_prefixes = [
        '10.',
        '172.16.', '172.17.', '172.18.', '172.19.',
        '172.20.', '172.21.', '172.22.', '172.23.',
        '172.24.', '172.25.', '172.26.', '172.27.',
        '172.28.', '172.29.', '172.30.', '172.31.',
        '192.168.'
    ]

    src_is_private = any(src_ip.startswith(prefix) for prefix in private_ip_prefixes)
    dst_is_private = any(dst_ip.startswith(prefix) for prefix in private_ip_prefixes)

    if src_is_private and not dst_is_private:
        return 'outbound'
    elif not src_is_private and dst_is_private:
        return 'inbound'
    else:
        return 'internal'


def process_directory(directory_path, tshark_path):
    """Process all PCAP files in a directory"""
    success_count = 0
    failed_count = 0

    # Get list of all PCAP files
    pcap_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(root, file))

    total_files = len(pcap_files)
    print(f"\nFound {total_files} PCAP files to process")

    # Process each file
    for index, pcap_path in enumerate(pcap_files, 1):
        print(f"\nProcessing file {index}/{total_files}: {pcap_path}")
        if process_pcap(pcap_path, tshark_path):
            success_count += 1
        else:
            failed_count += 1

    print(f"\nProcessing complete:")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {failed_count} files")
    return success_count, failed_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract features from PCAP files')
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing PCAP files or path to single PCAP file')
    args = parser.parse_args()

    # Find tshark
    tshark_path = find_tshark()
    print(f"Using tshark from: {tshark_path}")

    start_time = time.time()
    input_path = args.input

    if os.path.isfile(input_path):
        print(f"\nProcessing single file: {input_path}")
        process_pcap(input_path, tshark_path)
    elif os.path.isdir(input_path):
        print(f"\nProcessing all PCAP files in directory: {input_path}")
        success_count, failed_count = process_directory(input_path, tshark_path)
    else:
        print(f"Error: {input_path} is not a valid file or directory")

    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")

################ USing dpkt ############################
# import dpkt
# import socket
# import os
# import csv
# import time
# from datetime import datetime
#
# PROTO_DICT = {dpkt.tcp.TCP: "TCP", dpkt.udp.UDP: "UDP"}
#
#
# def inet_to_str(inet):
#     """Convert inet object to a string"""
#     try:
#         return socket.inet_ntop(socket.AF_INET, inet)
#     except ValueError:
#         return socket.inet_ntop(socket.AF_INET6, inet)
#
#
# def process_pcap(pcap_path):
#     """Process a PCAP file and extract features"""
#     base_path = os.path.splitext(pcap_path)[0]
#     txt_path = f"{base_path}_features.txt"
#     csv_path = f"{base_path}_features.csv"
#     counter = 0
#     packets = []
#
#     try:
#         with open(pcap_path, 'rb') as f:
#             pcap = dpkt.pcap.Reader(f)
#
#             # Process each packet
#             for ts, buf in pcap:
#                 try:
#                     # Unpack the Ethernet frame
#                     eth = dpkt.ethernet.Ethernet(buf)
#
#                     # Make sure the Ethernet frame contains an IP packet
#                     if isinstance(eth.data, dpkt.ip.IP):
#                         ip = eth.data
#                     elif isinstance(eth.data, str):
#                         try:
#                             ip = dpkt.ip.IP(buf)
#                         except dpkt.UnpackError:
#                             continue
#                     else:
#                         continue
#
#                     # Get the protocol
#                     proto = ip.data
#
#                     if type(ip.data) in PROTO_DICT:
#                         # Extract features using exactly your method
#                         size = len(ip)  # IP packet size (excluding Ethernet header)
#
#                         packet_info = {
#                             'Timestamp': ts,
#                             'SourceIP': inet_to_str(ip.src),
#                             'SourcePort': proto.sport,
#                             'DestinationIP': inet_to_str(ip.dst),
#                             'DestinationPort': proto.dport,
#                             'Protocol': PROTO_DICT[type(ip.data)],
#                             'Size': size
#                         }
#                         packets.append(packet_info)
#                         counter += 1
#
#                 except Exception as e:
#                     print(f"Error processing packet: {e}")
#                     continue
#
#         # Write to text file
#         with open(txt_path, 'w', encoding='utf-8') as txt_file:
#             for packet in packets:
#                 txt_file.write(
#                     f"Timestamp: {packet['Timestamp']}, "
#                     f"SourceIP: {packet['SourceIP']}, "
#                     f"SourcePort: {packet['SourcePort']}, "
#                     f"DestinationIP: {packet['DestinationIP']}, "
#                     f"DestinationPort: {packet['DestinationPort']}, "
#                     f"Protocol: {packet['Protocol']}, "
#                     f"Size: {packet['Size']}\n"
#                 )
#
#         # Write to CSV file
#         with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#             fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
#                           'DestinationPort', 'Protocol', 'Size']
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(packets)
#
#         print(f"Successfully processed {counter} packets from {pcap_path}")
#         print(f"Output written to:\n{txt_path}\n{csv_path}")
#         return True
#
#     except Exception as e:
#         print(f"Error processing PCAP file {pcap_path}: {str(e)}")
#         return False
#
#
# def process_directory(directory_path):
#     """Process all PCAP files in a directory"""
#     success_count = 0
#     failed_count = 0
#
#     # Get list of all PCAP files
#     pcap_files = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(('.pcap', '.pcapng')):
#                 pcap_files.append(os.path.join(root, file))
#
#     total_files = len(pcap_files)
#     print(f"\nFound {total_files} PCAP files to process")
#
#     # Process each file
#     for index, pcap_path in enumerate(pcap_files, 1):
#         print(f"\nProcessing file {index}/{total_files}: {pcap_path}")
#         if process_pcap(pcap_path):
#             success_count += 1
#         else:
#             failed_count += 1
#
#     print(f"\nProcessing complete:")
#     print(f"Successfully processed: {success_count} files")
#     print(f"Failed to process: {failed_count} files")
#     return success_count, failed_count
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Extract features from PCAP files')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Directory containing PCAP files or path to single PCAP file')
#     args = parser.parse_args()
#
#     start_time = time.time()
#     input_path = args.input
#
#     if os.path.isfile(input_path):
#         print(f"\nProcessing single file: {input_path}")
#         process_pcap(input_path)
#     elif os.path.isdir(input_path):
#         print(f"\nProcessing all PCAP files in directory: {input_path}")
#         success_count, failed_count = process_directory(input_path)
#     else:
#         print(f"Error: {input_path} is not a valid file or directory")
#
#     total_time = time.time() - start_time
#     print(f"\nTotal processing time: {total_time:.2f} seconds")


##################### Updated tshark code with more features -- slow on voip and file transfer ############################
# import os
# import csv
# import subprocess
# import time
# from datetime import datetime
# import statistics
# from collections import defaultdict
#
#
# def find_tshark():
#     """
#     Find tshark executable in common installation paths
#     For this you need to install wire shark and add it to the path, and then check you have tshark in the path
#     Usually it is in the path C:\Program Files\Wireshark\tshark.exe
#     """
#     common_paths = [
#         r"C:\Program Files\Wireshark\tshark.exe",
#         r"C:\Program Files (x86)\Wireshark\tshark.exe",
#         "/usr/bin/tshark",
#         "/usr/local/bin/tshark"
#     ]
#     for path in common_paths:
#         if os.path.exists(path):
#             return path
#     return "tshark"
#
#
# class FlowStats:
#     def __init__(self, first_packet_time):
#         self.fwd_packets = []
#         self.bwd_packets = []
#         self.fin_count = 0
#         self.syn_count = 0
#         self.rst_count = 0
#         self.psh_count = 0
#         self.ack_count = 0
#         self.first_packet_time = first_packet_time
#         self.last_fwd_time = None
#         self.last_bwd_time = None
#
#     def add_packet(self, timestamp, size, flags, is_forward):
#         if is_forward:
#             self.fwd_packets.append((float(timestamp), int(size)))
#             self.last_fwd_time = float(timestamp)
#         else:
#             self.bwd_packets.append((float(timestamp), int(size)))
#             self.last_bwd_time = float(timestamp)
#
#         if flags:
#             if 'F' in flags: self.fin_count += 1
#             if 'S' in flags: self.syn_count += 1
#             if 'R' in flags: self.rst_count += 1
#             if 'P' in flags: self.psh_count += 1
#             if 'A' in flags: self.ack_count += 1
#
#     def calculate_stats(self):
#         stats = {}
#
#         # Basic packet counts and lengths
#         stats['fwd_packets_amount'] = len(self.fwd_packets)
#         stats['bwd_packets_amount'] = len(self.bwd_packets)
#         fwd_lengths = [p[1] for p in self.fwd_packets]
#         bwd_lengths = [p[1] for p in self.bwd_packets]
#         stats['fwd_packets_length'] = sum(fwd_lengths) if fwd_lengths else 0
#         stats['bwd_packets_length'] = sum(bwd_lengths) if bwd_lengths else 0
#
#         # Packet size statistics
#         stats['max_fwd_packet'] = max(fwd_lengths) if fwd_lengths else 0
#         stats['min_fwd_packet'] = min(fwd_lengths) if fwd_lengths else 0
#         stats['max_bwd_packet'] = max(bwd_lengths) if bwd_lengths else 0
#         stats['min_bwd_packet'] = min(bwd_lengths) if bwd_lengths else 0
#
#         all_lengths = fwd_lengths + bwd_lengths
#         stats['min_packet_size'] = min(all_lengths) if all_lengths else 0
#         stats['max_packet_size'] = max(all_lengths) if all_lengths else 0
#         stats['mean_packet_size'] = statistics.mean(all_lengths) if all_lengths else 0
#         stats['STD_packet_size'] = statistics.stdev(all_lengths) if len(all_lengths) > 1 else 0
#
#         # Inter-arrival times
#         if len(self.fwd_packets) > 1:
#             fwd_times = [j[0] - i[0] for i, j in zip(self.fwd_packets[:-1], self.fwd_packets[1:])]
#             stats['min_fwd_inter_arrival_time'] = min(fwd_times)
#             stats['max_fwd_inter_arrival_time'] = max(fwd_times)
#             stats['mean_fwd_inter_arrival_time'] = statistics.mean(fwd_times)
#         else:
#             stats['min_fwd_inter_arrival_time'] = 0
#             stats['max_fwd_inter_arrival_time'] = 0
#             stats['mean_fwd_inter_arrival_time'] = 0
#
#         if len(self.bwd_packets) > 1:
#             bwd_times = [j[0] - i[0] for i, j in zip(self.bwd_packets[:-1], self.bwd_packets[1:])]
#             stats['min_bwd_inter_arrival_time'] = min(bwd_times)
#             stats['max_bwd_inter_arrival_time'] = max(bwd_times)
#             stats['mean_bwd_inter_arrival_time'] = statistics.mean(bwd_times)
#         else:
#             stats['min_bwd_inter_arrival_time'] = 0
#             stats['max_bwd_inter_arrival_time'] = 0
#             stats['mean_bwd_inter_arrival_time'] = 0
#
#         # Packets per second
#         duration = (max(self.last_fwd_time,
#                         self.last_bwd_time) - self.first_packet_time) if self.last_fwd_time and self.last_bwd_time else 1
#         if duration > 0:
#             stats['pps_fwd'] = len(self.fwd_packets) / duration
#             stats['pps_bwd'] = len(self.bwd_packets) / duration
#         else:
#             stats['pps_fwd'] = len(self.fwd_packets)
#             stats['pps_bwd'] = len(self.bwd_packets)
#
#         # TCP flags
#         stats['FIN_count'] = self.fin_count
#         stats['SYN_count'] = self.syn_count
#         stats['RST_count'] = self.rst_count
#         stats['PSH_count'] = self.psh_count
#         stats['ACK_count'] = self.ack_count
#
#         return stats
#
#
# def process_pcap(pcap_path, tshark_path):
#     """Process a PCAP file and combine packet and flow statistics"""
#     base_path = os.path.splitext(pcap_path)[0]
#     csv_path = f"{base_path}_features.csv"
#     txt_path = f"{base_path}_features.txt"
#
#     try:
#         cmd = [
#             tshark_path,
#             '-r', pcap_path,
#             '-T', 'fields',
#             '-E', 'separator=,',
#             '-e', 'frame.time_epoch',
#             '-e', 'ip.src',
#             '-e', 'tcp.srcport',
#             '-e', 'udp.srcport',
#             '-e', 'ip.dst',
#             '-e', 'tcp.dstport',
#             '-e', 'udp.dstport',
#             '-e', 'ip.len',
#             '-e', 'tcp.flags',
#             '-e', 'ip.proto',
#             '-Y', 'ip'
#         ]
#
#         print(f"Processing: {pcap_path}")
#
#         result = subprocess.run(cmd, capture_output=True, text=True)
#         if result.returncode != 0:
#             print(f"Error processing file: {result.stderr}")
#             return False
#
#         # Process packets and collect flow information
#         flows = defaultdict(lambda: None)
#         combined_records = []
#
#         for line in result.stdout.strip().split('\n'):
#             if not line.strip():
#                 continue
#
#             fields = line.strip().split(',')
#             if len(fields) < 9:
#                 continue
#
#             try:
#                 timestamp = float(fields[0])
#                 src_ip = fields[1]
#                 src_port = fields[2] or fields[3]
#                 dst_ip = fields[4]
#                 dst_port = fields[5] or fields[6]
#                 size = int(fields[7]) if fields[7] else 0
#                 flags = fields[8]
#
#                 # Determine flow direction
#                 forward_key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
#                 backward_key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"
#
#                 if forward_key in flows:
#                     flow_key = forward_key
#                     is_forward = True
#                 elif backward_key in flows:
#                     flow_key = backward_key
#                     is_forward = False
#                 else:
#                     flow_key = forward_key
#                     is_forward = True
#                     flows[flow_key] = FlowStats(timestamp)
#
#                 # Add to flow statistics
#                 flows[flow_key].add_packet(timestamp, size, flags, is_forward)
#
#                 # Create combined record
#                 timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
#                 protocol = 'TCP' if fields[2] else 'UDP'
#                 direction = 'outbound' if is_forward else 'inbound'
#
#                 # Get flow statistics for this packet
#                 flow_stats = flows[flow_key].calculate_stats()
#
#                 # Combine packet and flow information
#                 record = {
#                     #'Timestamp': timestamp_str,
#                     'Timestamp': timestamp,  # Keep timestamp as float for sorting
#                     'SourceIP': src_ip,
#                     'SourcePort': src_port,
#                     'DestinationIP': dst_ip,
#                     'DestinationPort': dst_port,
#                     'Protocol': protocol,
#                     'Size': size,
#                     'Direction': direction,
#                     **flow_stats  # Add all flow statistics
#                 }
#
#                 combined_records.append(record)
#
#             except Exception as e:
#                 print(f"Error processing packet: {e}")
#                 continue
#
#         if combined_records:
#             # Define field order
#             fieldnames = ['Timestamp', 'SourceIP', 'SourcePort', 'DestinationIP',
#                           'DestinationPort', 'Protocol', 'Size', 'Direction',
#                           'fwd_packets_amount', 'bwd_packets_amount',
#                           'fwd_packets_length', 'bwd_packets_length',
#                           'max_fwd_packet', 'min_fwd_packet',
#                           'max_bwd_packet', 'min_bwd_packet',
#                           'min_fwd_inter_arrival_time', 'max_fwd_inter_arrival_time',
#                           'mean_fwd_inter_arrival_time', 'min_bwd_inter_arrival_time',
#                           'max_bwd_inter_arrival_time', 'mean_bwd_inter_arrival_time',
#                           'FIN_count', 'SYN_count', 'RST_count', 'PSH_count',
#                           'ACK_count', 'min_packet_size', 'max_packet_size',
#                           'mean_packet_size', 'STD_packet_size',
#                           'pps_fwd', 'pps_bwd']
#
#             # Write CSV
#             with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
#                 writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#                 writer.writeheader()
#                 writer.writerows(combined_records)
#
#             # Write TXT
#             with open(txt_path, 'w', encoding='utf-8') as txt_file:
#                 # Write header
#                 txt_file.write(", ".join(fieldnames) + "\n")
#
#                 # Write records
#                 for record in combined_records:
#                     values = [str(record.get(field, '')) for field in fieldnames]
#                     txt_file.write(", ".join(values) + "\n")
#
#             print(f"Processed {len(combined_records)} packets")
#             print(f"Output written to:\n{csv_path}\n{txt_path}")
#             return True
#
#         else:
#             print("No valid packets found in the PCAP file")
#             return False
#
#     except Exception as e:
#         print(f"Error processing PCAP file {pcap_path}: {str(e)}")
#         return False
#
#
# def process_directory(directory_path, tshark_path):
#     """Process all PCAP files in a directory"""
#     success_count = 0
#     failed_count = 0
#
#     pcap_files = []
#     for root, _, files in os.walk(directory_path):
#         for file in files:
#             if file.endswith(('.pcap', '.pcapng')):
#                 pcap_files.append(os.path.join(root, file))
#
#     total_files = len(pcap_files)
#     print(f"\nFound {total_files} PCAP files to process")
#
#     for index, pcap_path in enumerate(pcap_files, 1):
#         print(f"\nProcessing file {index}/{total_files}: {pcap_path}")
#         if process_pcap(pcap_path, tshark_path):
#             success_count += 1
#         else:
#             failed_count += 1
#
#     print(f"\nProcessing complete:")
#     print(f"Successfully processed: {success_count} files")
#     print(f"Failed to process: {failed_count} files")
#     return success_count, failed_count
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Extract combined features from PCAP files')
#     parser.add_argument('--input', type=str, required=True,
#                         help='Directory containing PCAP files or path to single PCAP file')
#     args = parser.parse_args()
#
#     tshark_path = find_tshark()
#     print(f"Using tshark from: {tshark_path}")
#
#     start_time = time.time()
#     input_path = args.input
#
#     if os.path.isfile(input_path):
#         print(f"\nProcessing single file: {input_path}")
#         process_pcap(input_path, tshark_path)
#     elif os.path.isdir(input_path):
#         print(f"\nProcessing all PCAP files in directory: {input_path}")
#         success_count, failed_count = process_directory(input_path, tshark_path)
#     else:
#         print(f"Error: {input_path} is not a valid file or directory")
#
#     total_time = time.time() - start_time
#     print(f"\nTotal processing time: {total_time:.2f} seconds")