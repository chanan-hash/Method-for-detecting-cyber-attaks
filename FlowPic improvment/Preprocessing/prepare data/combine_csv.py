import os
import csv
import argparse

def combine_csv(input_dir):
    # List all CSV files in the specified directory
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]
    csv_name = 'combined_file_dataset.csv' # can be changed according to the dataset type
    # Open the output file in write mode
    with open(csv_name, 'w', newline='') as outfile: # can be changed according to the dataset type 
        writer = csv.writer(outfile)
        header_written = False

        # Iterate through each CSV file
        for file in csv_files:
            with open(os.path.join(input_dir, file), 'r') as infile:
                reader = csv.reader(infile)
                header = next(reader)

                # Add the new column to the header
                if not header_written:
                    header.append('Attribution')
                    writer.writerow(header)
                    header_written = True

                # Write the rest of the rows with the new column
                for row in reader:
                    row.append('file') # can be changed according to the dataset type
                    writer.writerow(row)

#    print("All CSV files have been combined into 'combined_video_voip_dataset.csv'.")
    print(f"All CSV files have been combined into '{csv_name}'.") # can be changed according to the dataset type

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine CSV files from a directory and add an Attribution column.')
    parser.add_argument('--input', type=str, required=True, help='Directory containing CSV files')
    args = parser.parse_args()

    combine_csv(args.input)

# import os
# import pandas as pd
# import argparse

# def combine_csv_files(input_directory, output_file):
#     # List all CSV files in the directory
#     csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

#     if not csv_files:
#         print("No CSV files found in the directory.")
#         return

#     combined_df = pd.DataFrame()

#     for csv_file in csv_files:
#         file_path = os.path.join(input_directory, csv_file)
#         # Read the CSV file
#         df = pd.read_csv(file_path)
#         # Add the 'attribution' column
#         df['Attribution'] = 'file'
#         # Append to the combined DataFrame
#         combined_df = pd.concat([combined_df, df], ignore_index=True)

#     # Save the combined DataFrame to the output file
#     combined_df.to_csv(output_file, index=False)
#     print(f"Combined CSV file saved as {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Combine all CSV files in a directory into one CSV file.")
#     parser.add_argument('--input', required=True, help="Path to the directory containing CSV files.")
#     parser.add_argument('--output', default="combined.csv", help="Path to the output CSV file (default: combined.csv).")

#     args = parser.parse_args()

#     combine_csv_files(args.input, args.output)

# # python combine_csv_files.py --input "data" --output "result.csv"
 