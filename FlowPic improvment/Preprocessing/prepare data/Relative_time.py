import os
import pandas as pd
import argparse

def add_relative_time(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            
            if 'Timestamp' in df.columns:
                first_timestamp = df['Timestamp'].iloc[0]
                df['RelativeTime'] = df['Timestamp'] - first_timestamp
                df.to_csv(filepath, index=False)
            else:
                print(f"No 'Timestamp' column in {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add relative time column to csv files')
    parser.add_argument('--input', type=str,required=True ,help='Directory containing csv files')
    args = parser.parse_args()
    add_relative_time(args.input)
