# import os
# import pandas as pd
# import glob
# import re


# def process_csv_range(directory_path, start_num, end_num, new_column_name, column_value, file_count=None):
#     """
#     Process CSV files in a directory within a specified range, adding a new column with a given value.
#     Updates the files in place rather than creating new ones.

#     Args:
#         directory_path (str): Path to the directory containing CSV files
#         start_num (int): Start of range (inclusive)
#         end_num (int): End of range (inclusive)
#         new_column_name (str): Name of the new column to add
#         column_value: Value to fill the new column with
#         file_count (int, optional): Limit the number of files to process
#     """
#     try:
#         # Ensure directory exists
#         if not os.path.exists(directory_path):
#             raise ValueError(f"Directory '{directory_path}' does not exist")

#         # Get all CSV files in the directory
#         csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

#         if not csv_files:
#             raise ValueError(f"No CSV files found in directory '{directory_path}'")

#         # Function to extract number from filename
#         def extract_number(filename):
#             numbers = re.findall(r'\d+', os.path.basename(filename))
#             return int(numbers[0]) if numbers else -1

#         # Filter and sort files based on the range
#         filtered_files = sorted([
#             f for f in csv_files
#             if start_num <= extract_number(f) <= end_num
#         ], key=extract_number)

#         if not filtered_files:
#             raise ValueError(f"No CSV files found within range {start_num}-{end_num}")

#         # Limit the number of files if specified
#         if file_count is not None:
#             filtered_files = filtered_files[:file_count]

#         # Process each file
#         for file_path in filtered_files:
#             try:
#                 # Read the CSV
#                 df = pd.read_csv(file_path)

#                 # Add new column with specified value
#                 df[new_column_name] = column_value

#                 # Save back to the same file
#                 df.to_csv(file_path, index=False)
#                 print(f"Updated {os.path.basename(file_path)} with new column '{new_column_name}' = {column_value}")

#             except Exception as e:
#                 print(f"Error processing {file_path}: {str(e)}")
#                 continue

#         print(f"\nSuccessfully processed {len(filtered_files)} files")

#     except Exception as e:
#         print(f"Error: {str(e)}")


# # Example usage
# if __name__ == "__main__":
#     directory = r"C:\Users\חנן\Desktop\אריאל אונ'\שנה ג\זיהוי התקפות\FlowPic\ISCX2016\iscx_chat_vpn.raw"  # Replace with your directory path

#     # Process different ranges with different values
#     # Format: (start, end, column_name, value, file_count)
#     # Process different ranges with different values
#     ranges_to_process = [
#         # (start, end, column_name, value)
#         # (1, 3, "App", "vpn aim",3),
#         (4, 16, "App", "vpn facebook",13),
#         # (17, 30, "App", "vpn hangouts",14),
#         # (31, 34, "App", "vpn icq",4),
#         # (35, 45, "App", "vpn skype",11)
#         # Add more ranges as needed
#     ]

#     # Process each range
#     for start, end, column, value, count in ranges_to_process:
#         print(f"\nProcessing files {start}-{end} with value '{value}'")
#         process_csv_range(directory, start, end, column, value, count)

# #
# # # Example usage
# # if __name__ == "__main__":
# #     # Configuration
# #     directory = r"C:\Users\חנן\Desktop\אריאל אונ'\שנה ג\זיהוי התקפות\FlowPic\ISCX2016\iscx_chat_vpn.raw"  # Replace with your directory path
# #     file_range = (0, 3)  # Process files with numbers 1-10 in their names
# #     new_column = "App"  # Name of the new column
# #     value = "vpn aim"  # Value to fill the new column with
# #
# #     # Run the processor
# #     process_csv_files(directory, file_range, new_column, value)

import os
import glob
import pandas as pd
import argparse

def add_direction_column(directory_path):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    for file_path in csv_files:
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            if "SourceIP" not in df.columns or "DestinationIP" not in df.columns:
                print(f"Skipping {os.path.basename(file_path)}: missing 'SourceIP' or 'DestinationIP' columns.")
                continue

            # Assume each CSV contains only 2 unique IP addresses
            unique_ips = pd.unique(df[['SourceIP', 'DestinationIP']].values.ravel())
            if len(unique_ips) != 2:
                print(f"{os.path.basename(file_path)} does not have exactly 2 unique IPs. Setting Direction=-1 for all rows.")
                df["Direction"] = -1
            else:
                # The first row's SourceIP is assumed to be the client
                client_ip = df.iloc[0]["SourceIP"]
                # The other IP is assumed to be the server
                server_ip = unique_ips[0] if unique_ips[0] != client_ip else unique_ips[1]

                # Define a function to compute the direction per row
                def compute_direction(row):
                    if row["SourceIP"] == client_ip and row["DestinationIP"] == server_ip:
                        return 1  # Client to server
                    elif row["SourceIP"] == server_ip and row["DestinationIP"] == client_ip:
                        return 0  # Server to client
                    else:
                        return -1

                df["Direction"] = df.apply(compute_direction, axis=1)
            
            # Save the updated CSV (overwrites the original file)
            df.to_csv(file_path, index=False)
            print(f"Updated {os.path.basename(file_path)} with new column 'Direction'.")

            # Check if there is any row with Direction = -1
            if (df["Direction"] == -1).any():
                print(f"WARNING: {os.path.basename(file_path)} contains rows with -1 in the 'Direction' column.")

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add 'Direction' column to CSV files based on SourceIP and DestinationIP")
    parser.add_argument("--input", type=str, required=True, help="Directory containing CSV files")
    args = parser.parse_args()

    add_direction_column(args.input)