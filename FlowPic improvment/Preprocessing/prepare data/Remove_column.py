import os
import pandas as pd
import glob
import argparse


def remove_column_from_csvs(directory_path, column_name):
    """
    Remove a specific column from all CSV files in the directory.

    Args:
        directory_path (str): Path to the directory containing CSV files
        column_name (str): Name of the column to remove
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory '{directory_path}' does not exist")

        # Get all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in directory '{directory_path}'")

        # Process each file
        for file_path in csv_files:
            try:
                # Read the CSV
                df = pd.read_csv(file_path)

                # Check if column exists
                if column_name in df.columns:
                    # Remove the column
                    df = df.drop(columns=[column_name])

                    # Save back to the same file
                    df.to_csv(file_path, index=False)
                    print(f"Removed column '{column_name}' from {os.path.basename(file_path)}")
                else:
                    print(f"Column '{column_name}' not found in {os.path.basename(file_path)}")

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue

        print("\nColumn removal completed!")

    except Exception as e:
        print(f"Error: {str(e)}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove a specific column from all CSV files in the directory")
    parser.add_argument("--input", type=str, help="Path to the directory containing CSV files")
    parser.add_argument("--column", type=str, help="Name of the column to remove")
    args = parser.parse_args()
    
    remove_column_from_csvs(args.input, args.column)