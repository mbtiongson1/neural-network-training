import csv
import os
from pathlib import Path

def loadCSV(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            print(f'  Columns: {reader.fieldnames}')
        for row in reader:
            cleaned = {k.strip() if k else None: v for k, v in row.items()}
            cleaned = {k: v for k, v in cleaned.items() if k is not None}
            data.append(cleaned)
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
            cleaned = {k: v for k, v in row.items() if k is not None}
            cleaned['network'] = network
            alldata.append(cleaned)
        
        print(f'Loaded {len(rows)} rows from {network}')
    
    if alldata:
        allkeys = set()
        for row in alldata:
            allkeys.update(row.keys())
        
        fieldnames = ['network'] + sorted([k for k in allkeys if k and k != 'network'])
        
        with open(outputfile, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(alldata)
        
        print(f'\nCombined {len(alldata)} total rows written to {outputfile}')
        
        outputFinal(alldata)
        rankScores(alldata)
    else:
        print('No data to write')

def outputFinal(data):
    os.makedirs('final', exist_ok=True)
    
    allkeys = set()
    for row in data:
        allkeys.update(row.keys())
    
    fieldnames = ['network'] + sorted([k for k in allkeys if k and k != 'network'])
    
    fullpath = os.path.join('final', 'full_scores.csv')
    with open(fullpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f'Full scores written to {fullpath}')
    
    valid = [row for row in data if row.get('f1_macro') and row['f1_macro'] != '']
    for row in valid:
        row['f1_macro'] = float(row['f1_macro'])
    
    best = max(valid, key=lambda x: x['f1_macro'])
    
    bestpath = os.path.join('final', 'scores.csv')
    with open(bestpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best)
    
    print(f'Best score written to {bestpath} (Network: {best["network"]}, F1: {best["f1_macro"]:.6f})')
    
    printFullScores(data)

def printFullScores(data):
    valid = [row for row in data if row.get('f1_macro') and row['f1_macro'] != '']
    
    numcols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'matthews']
    
    for row in valid:
        row['f1_macro'] = float(row['f1_macro'])
        for col in numcols:
            if col in row and row[col]:
                try:
                    row[col] = float(row[col])
                except:
                    pass
        timekey = None
        for k in row.keys():
            if 'time' in k.lower():
                timekey = k
                break
        if timekey and row.get(timekey):
            try:
                row['total_time'] = float(row[timekey])
            except:
                row['total_time'] = 0.0
        else:
            row['total_time'] = 0.0
    
    ranked = sorted(valid, key=lambda x: x['f1_macro'], reverse=True)
    
    print('\n# Full Scores\n')
    
    cols = ['Rank', 'network', 'iteration', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'matthews', 'total_time']
    
    header = '| ' + ' | '.join(cols) + ' |'
    separator = '|' + '|'.join(['---'] * len(cols)) + '|'
    
    print(header)
    print(separator)
    
    for i, row in enumerate(ranked, 1):
        values = [str(i)]
        for col in cols[1:]:
            val = row.get(col, '')
            if isinstance(val, float):
                values.append(f'{val:.5f}')
            else:
                values.append(str(val))
        print('| ' + ' | '.join(values) + ' |')

def rankScores(data):
    valid = [row for row in data if row.get('f1_macro') and row['f1_macro'] != '']
    
    for row in valid:
        row['f1_macro'] = float(row['f1_macro'])
        timekey = None
        for k in row.keys():
            if 'time' in k.lower():
                timekey = k
                break
        if timekey and row.get(timekey):
            row['total_time'] = float(row[timekey])
        else:
            row['total_time'] = 0.0
    
    ranked = sorted(valid, key=lambda x: x['f1_macro'], reverse=True)
    
    print('\n# Full Scores\n')
    
    allkeys = set()
    for row in ranked:
        allkeys.update(row.keys())
    
    cols = ['Rank', 'network', 'iteration', 'f1_macro', 'total_time']
    header = '| ' + ' | '.join(cols) + ' |'
    separator = '|' + '|'.join(['---'] * len(cols)) + '|'
    
    print(header)
    print(separator)
    
    for i, row in enumerate(ranked, 1):
        network = row['network']
        iteration = row['iteration']
        f1 = row['f1_macro']
        totaltime = row['total_time']
        print(f'| {i} | {network} | {iteration} | {f1:.5f} | {totaltime:.5f} |')

if __name__ == '__main__':
    exportdir = 'export'
    outputfile = 'combined_scores.csv'
    
    combineScores(exportdir, outputfile)