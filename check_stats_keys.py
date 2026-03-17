import json
import os
from config_01 import SYMPTOM_STATS_PATH

def main():
    print(f"Reading file: {SYMPTOM_STATS_PATH}")

    if not os.path.exists(SYMPTOM_STATS_PATH):
        print("Error: File not found. Please run 02_extract_stats.py first.")
        return

    try:
        with open(SYMPTOM_STATS_PATH, 'r') as f:
            data = json.load(f)

        # Get all keys and sort them alphabetically for easier reading
        keys = sorted(data.keys())

        print(f"\nFound {len(keys)} unique diseases in statistics:\n")
        print("-" * 50)
        for key in keys:
            print(f"'{key}'")
        print("-" * 50)
        print(f"Total count: {len(keys)}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()