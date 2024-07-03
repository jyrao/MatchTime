from score_single import calculate_metrics

import os
import glob
import pandas as pd
from tqdm import tqdm

def process_all_csv(folder_path, output_folder):

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    results = []

    for csv_file in tqdm(csv_files, desc="Processing CSV files"):
        metrics = calculate_metrics(csv_file)
        row = [os.path.basename(csv_file)]
        row.extend(metrics.values())
        results.append(row)

    column_names = ['filename']
    if results:
        column_names.extend(metrics.keys())

    df = pd.DataFrame(results, columns=column_names)
    basename = os.path.basename(folder_path) + '_scores.csv'
    output_file = os.path.join(output_folder, basename)

    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files for football match commentary.')
    parser.add_argument('--folder_path', type=str, default='./inference_result', help='Path to the folder containing CSV files')
    parser.add_argument('--output_folder', type=str, default='./inference_result', help='Output folder for processed results')

    args = parser.parse_args()

    process_all_csv(args.folder_path, args.output_folder)