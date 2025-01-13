import os
import csv
import argparse

def combine_csv(input_dir):
    # List all CSV files in the specified directory
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('.csv')]

    # Open the output file in write mode
    with open('combined_chat_dataset.csv', 'w', newline='') as outfile:
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
                    row.append('chat')
                    writer.writerow(row)

    print("All CSV files have been combined into 'combined_chat_dataset.csv'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine CSV files from a directory and add an Attribution column.')
    parser.add_argument('--input', type=str, required=True, help='Directory containing CSV files')
    args = parser.parse_args()

    combine_csv(args.input)