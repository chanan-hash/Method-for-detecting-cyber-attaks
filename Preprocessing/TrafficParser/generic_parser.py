#!/usr/bin/env python
"""
Use DPKT to read in a pcap file and create one directional sessions of packets sizes (ip total length) and ts.

From https://github.com/talshapira/FlowPic/tree/main repo, of talshapira, I made a little changes
"""
import dpkt
import os
import socket
import argparse
import csv
import time
import subprocess

FLAGS = None
# INPUT = "../dataset/iscxNTVPN2016/CompletePCAPs"#"../dataset/CICNTTor2017/Pcaps/tor" #"../dataset/iscxNTVPN2016/CompletePCAPs"#"./test_pacaps"#"../dataset/iscxNTVPN2016/CompletePCAPs" # ""
INPUT = './test_pcaps/my_chat'
FILTER_LIST = None # [(["audio", "voip"], True), (["vpn", "tor"], False)]

PROTO_DICT = {dpkt.tcp.TCP: "TCP", dpkt.udp.UDP: "UDP"}


def inet_to_str(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def get_pcaps_list(dir_path, filter_list=None):
    def filter_list_func(fn):
        if filter_list is not None:
            for filter_str_list, type in filter_list:
                result = any([filter_str in fn.lower() for filter_str in filter_str_list])
                if result is not type:
                    return False
        return True
    return [(os.path.join(dir_path, fn), fn) for fn in next(os.walk(dir_path))[2] if (".pcap" in os.path.splitext(fn)[-1] and filter_list_func(fn))]

# def parse_pcap(pcap, pcap_path, file_name):
#     """Print out information about each packet in a pcap
#        Args:
#            pcap: dpkt pcap reader object (dpkt.pcap.Reader)
#     """
#     counter = 0
#     pcap_dict = {}
#
#     # For each packet in the pcap process the contents
#     for ts, packet in pcap:
#
#         # Unpack the Ethernet frame
#         try:
#             eth = dpkt.ethernet.Ethernet(packet)
#         except dpkt.dpkt.NeedData:
#             print("dpkt.dpkt.NeedData")
#
#         # Make sure the Ethernet data contains an IP packet
#         if isinstance(eth.data, dpkt.ip.IP):
#             ip = eth.data
#         elif isinstance(eth.data, str):
#             try:
#                 ip = dpkt.ip.IP(packet)
#             except dpkt.UnpackError:
#                 continue
#         else:
#             continue
#
#         # Now unpack the data within the Ethernet frame (the IP packet)
#         # Pulling out src_ip, dst_ip, protocol (tcp/udp), dst/src port, length
#
#         proto = ip.data
#
#         # Print out the info
#         if type(ip.data)in PROTO_DICT:
#             session_tuple_key = (inet_to_str(ip.src), proto.sport, inet_to_str(ip.dst), proto.dport, PROTO_DICT[type(ip.data)])
#             pcap_dict.setdefault(session_tuple_key, (ts, [], []))
#             d = pcap_dict[session_tuple_key]
#             size = len(ip) #ip.len
#             d[1].append(round(ts - d[0], 6)), d[2].append(size)
#             counter += 1
#
#     print("Total Number of Parsed Packets in " + pcap_path + ": " + str(counter))
#
#     # csv_file_path = os.path.splitext(pcap_path)[0] + ".csv"
#     # with open(csv_file_path, 'wb') as csv_file:
#     #     writer = csv.writer(csv_file)
#     #     for key, value in pcap_dict.items():
#     #         writer.writerow([file_name.split(".")[0]] + list(key) + [value[0], len(value[1])] + value[1] + [None] + value[2])
#
#     # Chanan's update
#     csv_file_path = os.path.splitext(pcap_path)[0] + ".csv"
#     with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
#         writer = csv.writer(csv_file)
#         for key, value in pcap_dict.items():
#             writer.writerow([file_name.split(".")[0]] + list(key) + [value[0], len(value[1])] + value[1] + [None] + value[2])
#
#     # for k,v in pcap_dict.iteritems():
#     #     if len(v[1]) > 2000:
#     #         print(k, v[0], len(v[1]))
#     for k, v in pcap_dict.items():  # Use items() instead of iteritems()
#         if len(v[1]) > 2000:
#             print(k, v[0], len(v[1]))

# Also Chanan's update - added text file
def parse_pcap(pcap, pcap_path, file_name):
    counter = 0
    pcap_dict = {}

    text_file_path = os.path.splitext(pcap_path)[0] + ".txt"
    with open(text_file_path, 'w', encoding='utf-8') as text_file:
        for ts, packet in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(packet)
            except dpkt.dpkt.NeedData:
                print("dpkt.dpkt.NeedData")
                continue

            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
            elif isinstance(eth.data, str):
                try:
                    ip = dpkt.ip.IP(packet)
                except dpkt.UnpackError:
                    continue
            else:
                continue

            proto = ip.data

            if type(ip.data) in PROTO_DICT:
                session_tuple_key = (inet_to_str(ip.src), proto.sport, inet_to_str(ip.dst), proto.dport, PROTO_DICT[type(ip.data)])
                pcap_dict.setdefault(session_tuple_key, (ts, [], []))
                d = pcap_dict[session_tuple_key]
                size = len(ip)
                d[1].append(round(ts - d[0], 6)), d[2].append(size)
                counter += 1

                text_file.write(f"Timestamp: {ts}, SourceIP: {inet_to_str(ip.src)}, SourcePort: {proto.sport}, DestinationIP: {inet_to_str(ip.dst)}, DestinationPort: {proto.dport}, Protocol: {PROTO_DICT[type(ip.data)]}, Size: {size}\n")

    print("Total Number of Parsed Packets in " + pcap_path + ": " + str(counter))

    csv_file_path = os.path.splitext(pcap_path)[0] + ".csv"
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in pcap_dict.items():
            writer.writerow([file_name.split(".")[0]] + list(key) + [value[0], len(value[1])] + value[1] + [None] + value[2])

            # Printing the same data to the console
            print([file_name.split(".")[0]] + list(key) + [value[0], len(value[1])] + value[1] + [None] + value[2])

    for k, v in pcap_dict.items():
        if len(v[1]) > 2000:
            print(k, v[0], len(v[1]))

# def generic_parser(file_list):
#     """Open up a pcap file and create a output file containing all one-directional parsed sessions"""
#     for pcap_path, file_name in file_list:
#         try:
#             with open(pcap_path, 'rb') as f:
#                 pcap = dpkt.pcap.Reader(f)
#                 parse_pcap(pcap, pcap_path, file_name)
#
#         except ValueError:
#             new_pcap_file = os.path.splitext(pcap_path)[0] + "_new.pcap"
#             os.system("editcap -F libpcap -T ether " + pcap_path + " " + new_pcap_file)
#
#             with open(new_pcap_file, 'rb') as f:
#                 pcap = dpkt.pcap.Reader(f)
#                 parse_pcap(pcap, pcap_path, file_name)
#
#             os.remove(new_pcap_file)

def generic_parser(file_list):
    """Open up a pcap file and create a output file containing all one-directional parsed sessions"""
    # First try to find editcap in common Wireshark installation paths
    wireshark_paths = [
        r"C:\Program Files\Wireshark",
        r"C:\Program Files (x86)\Wireshark"
    ]
    editcap_path = None
    for path in wireshark_paths:
        potential_path = os.path.join(path, "editcap.exe")
        if os.path.exists(potential_path):
            editcap_path = f'"{potential_path}"'  # Quote the path to handle spaces
            break

    for pcap_path, file_name in file_list:
        try:
            with open(pcap_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                parse_pcap(pcap, pcap_path, file_name)

        except ValueError as e:
            print(f"Error reading PCAP file: {e}")
            if editcap_path is None:
                print("ERROR: editcap not found. Please install Wireshark and add it to your PATH")
                print("Or modify wireshark_paths in the code to point to your Wireshark installation")
                continue

            new_pcap_file = os.path.splitext(pcap_path)[0] + "_new.pcap"
            command = f'{editcap_path} -F libpcap -T ether "{pcap_path}" "{new_pcap_file}"'

            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error running editcap: {result.stderr}")
                    continue

                with open(new_pcap_file, 'rb') as f:
                    pcap = dpkt.pcap.Reader(f)
                    parse_pcap(pcap, pcap_path, file_name)

            except Exception as e:
                print(f"Error processing file {pcap_path}: {e}")
            finally:
                # Clean up the temporary file if it exists
                if os.path.exists(new_pcap_file):
                    try:
                        os.remove(new_pcap_file)
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file {new_pcap_file}: {e}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=INPUT, help='Path to pcap')

    FLAGS = parser.parse_args()
    file_list = get_pcaps_list(FLAGS.input, FILTER_LIST)
    start_time = time.time()
    generic_parser(file_list)
    total_time = time.time() - start_time
    print("--- %s seconds ---" % total_time)
