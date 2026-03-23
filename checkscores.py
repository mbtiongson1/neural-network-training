import csv
import os
from pathlib import Path

def loadCSV(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
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
            row['network'] = network
            alldata.append(row)
        
        print(f'Loaded {len(rows)} rows from {network}')
    
    if alldata:
        fieldnames = ['network'] + list(alldata[0].keys())
        fieldnames.remove('network')
        fieldnames.insert(0, 'network')
        
        with open(outputfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(alldata)
        
        print(f'\nCombined {len(alldata)} total rows written to {outputfile}')
        
        rankScores(alldata)
    else:
        print('No data to write')

def rankScores(data):
    valid = [row for row in data if row.get('f1_macro') and row['f1_macro'] != '']
    
    for row in valid:
        row['f1_macro'] = float(row['f1_macro'])
    
    ranked = sorted(valid, key=lambda x: x['f1_macro'], reverse=True)
    
    print('\n' + '='*60)
    print('TOP 10 F1 SCORES (f1_macro)')
    print('='*60)
    
    for i, row in enumerate(ranked[:10], 1):
        network = row['network']
        iteration = row['iteration']
        f1 = row['f1_macro']
        print(f'{i:2d}. Network: {network:15s} | Iteration: {iteration:10s} | F1: {f1:.6f}')

if __name__ == '__main__':
    exportdir = 'export'
    outputfile = 'combined_scores.csv'
    
    combineScores(exportdir, outputfile)