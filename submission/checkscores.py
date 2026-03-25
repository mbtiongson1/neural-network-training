import csv
import os
from pathlib import Path

def loadCSV(filepath):
    """Load a CSV as a list of lists, stripping whitespace."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            row = [v.strip() for v in row]
            data.append(row)
    return data

def findScoresFiles(rootdir):
    found = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        if 'scores.csv' in filenames:
            found.append(os.path.join(dirpath, 'scores.csv'))
    return found

def extractNetworkName(filepath):
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part == 'export' and i + 1 < len(parts):
            return parts[i + 1]
    return 'unknown'

def combineScores(rootdir, outputfile):
    files = findScoresFiles(rootdir)
    if not files:
        print(f'No scores.csv files found in {rootdir}')
        return

    alldata = []

    for filepath in files:
        network = extractNetworkName(filepath)
        rows = loadCSV(filepath)

        for row in rows:
            cleaned = {f'col{i}': v for i, v in enumerate(row)}
            cleaned['network'] = network
            # map the correct column to f1_macro
            try:
                cleaned['f1_macro'] = float(row[45])  # adjust index as needed
            except (IndexError, ValueError):
                cleaned['f1_macro'] = 0.0
            alldata.append(cleaned)

        print(f'Loaded {len(rows)} rows from {network}')

    if not alldata:
        print('No data to write')
        return

    # Write combined CSV
    allkeys = set()
    for row in alldata:
        allkeys.update(row.keys())

    fieldnames = ['network'] + sorted([k for k in allkeys if k != 'network'])
    with open(outputfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(alldata)

    print(f'\nCombined {len(alldata)} total rows written to {outputfile}')

    # Output final scores
    outputFinal(alldata)

def outputFinal(data):
    os.makedirs('final', exist_ok=True)
    allkeys = set()
    for row in data:
        allkeys.update(row.keys())

    fieldnames = ['network'] + sorted([k for k in allkeys if k != 'network'])
    fullpath = os.path.join('final', 'full_scores.csv')
    with open(fullpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f'Full scores written to {fullpath}')

    # Only rows with f1_macro > 0
    valid = [row for row in data if row.get('f1_macro', 0) > 0]
    if not valid:
        print("No valid rows with f1_macro > 0")
        return

    # Best row by f1_macro
    best = max(valid, key=lambda x: x['f1_macro'])
    bestpath = os.path.join('final', 'scores.csv')
    with open(bestpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best)
    print(f'Best score written to {bestpath} (Network: {best["network"]}, F1: {best["f1_macro"]:.6f})')

if __name__ == '__main__':
    exportdir = 'export'
    outputfile = 'combined_scores.csv'
    combineScores(exportdir, outputfile)