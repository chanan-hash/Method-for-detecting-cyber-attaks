"""
This script takes a text file and converts it to a CSV file.
Here in the FlowPic after we've parsed the pcap file, we have it also in csv and text
The text is more readable and the csv is more structured.
So we want to construct again the csv file from the text file, with the same headlines.

The txt file content should look like this, for example
Timestamp: 1733660914.502248, SourceIP: 10.102.3.42, SourcePort: 64080, DestinationIP: 140.82.112.22, DestinationPort: 443, Protocol: TCP, Size: 52

We got this txt file from the pcap file, and now we want to convert it to a csv file.
From generic_parser.py we've created the txt file
"""

import os
import csv
import argparse

def convert_txt_to_csv(directory):
    if not os.path.isdir(directory):
        print(f"Error: The directory {directory} does not exist.")
        return

    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_file_path = os.path.join(directory, filename)
            csv_file_path = os.path.splitext(txt_file_path)[0] + ".csv"
            print(f"Processing file: {txt_file_path}")

            with open(txt_file_path, 'r', encoding='utf-8') as txt_file, open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Timestamp", "Source IP", "Source port", "Destination IP", "Destination port", "Protocol", "Size"])

                for line in txt_file:
                    parts = line.strip().split(", ")
                    row = []
                    for part in parts:
                        if ": " in part:
                            row.append(part.split(": ")[1])
                        else:
                            row.append("")
                    csv_writer.writerow(row)
            print(f"Created CSV file: {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Path to the directory containing txt files')
    args = parser.parse_args()

    convert_txt_to_csv(args.input)