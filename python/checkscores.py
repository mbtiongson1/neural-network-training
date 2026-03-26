import csv
import os
from pathlib import Path

def loadCSV(filepath):
    data = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            data.append([v.strip() for v in row])
    return data

def findScoresFiles(rootdir):
    found = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        if 'scores.csv' in filenames:
            found.append(os.path.join(dirpath, 'scores.csv'))
    return found

def findErrorsFiles(rootdir):
    found = []
    for dirpath, dirnames, filenames in os.walk(rootdir):
        if 'errors.csv' in filenames:
            found.append(os.path.join(dirpath, 'errors.csv'))
    return found

def findConvergenceEpoch(filepath):
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_err = float(row.get('train_error', 1.0))
                val_err = float(row.get('val_error', 1.0))
                if train_err < 0.02 and val_err < 0.02:
                    return row.get('epoch', 'DNF')
        return 'DNF'
    except Exception as e:
        print(f'Error reading {filepath}: {e}')
        return 'DNF'

def extractNetworkName(filepath):
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part == 'export' and i + 1 < len(parts):
            return parts[i + 1]
    return 'unknown'

def combineScores(rootdir, outputfile):
    scoresfiles = findScoresFiles(rootdir)
    if not scoresfiles:
        print(f'No scores.csv files found in {rootdir}')
        return

    errorsfiles = findErrorsFiles(rootdir)
    convergenceMap = {}
    for errfile in errorsfiles:
        network = extractNetworkName(errfile)
        convergenceMap[network] = findConvergenceEpoch(errfile)

    alldata = []

    for filepath in scoresfiles:
        network = extractNetworkName(filepath)
        rows = loadCSV(filepath)

        if len(rows) < 2:
            print(f'Skipping {network}: not enough rows')
            continue

        headers = rows[0]

        for row in rows[1:]:
            if row[0] == 'Epoch':
                continue

            cleaned = {'network': network}
            for i, col in enumerate(headers):
                cleaned[col] = row[i] if i < len(row) else ''

            try:
                cleaned['f1_macro'] = float(cleaned.get('f1_macro', 0))
            except ValueError:
                cleaned['f1_macro'] = 0.0

            cleaned['Epoch'] = convergenceMap.get(network, 'DNF')

            alldata.append(cleaned)

        print(f'Loaded {len(rows) - 1} rows from {network}')

    if not alldata:
        print('No data to write')
        return

    allkeys = set()
    for row in alldata:
        allkeys.update(row.keys())

    fieldnames = ['network'] + sorted([k for k in allkeys if k != 'network'])
    with open(outputfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(alldata)

    print(f'\nCombined {len(alldata)} total rows written to {outputfile}')
    outputFinal(alldata, fieldnames)

def outputFinal(data, fieldnames):
    os.makedirs('final', exist_ok=True)

    fullpath = os.path.join('final', 'full_scores.csv')
    with open(fullpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f'Full scores written to {fullpath}')

    valid = [row for row in data if row.get('f1_macro', 0) > 0]
    if not valid:
        print('No valid rows with f1_macro > 0')
        return

    best = max(valid, key=lambda x: x['f1_macro'])
    bestpath = os.path.join('final', 'scores.csv')
    with open(bestpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best)
    print(f'Best score written to {bestpath} (Network: {best["network"]}, F1: {best["f1_macro"]:.6f})')

    generateResultsMarkdown(data, valid)

def generateResultsMarkdown(alldata, validdata):
    sorted_by_f1 = sorted(validdata, key=lambda x: float(x.get('f1_macro', 0)), reverse=True)

    ranked_table = "## Results\n\n### **Ranked by F1 Macro**\n\n"
    ranked_table += "| Rank | Network | Epoch (MSE<0.02) | f1\\_macro | Notes |\n"
    ranked_table += "| --- | --- | --- | --- | --- |\n"

    for i, row in enumerate(sorted_by_f1, 1):
        network = row.get('network', 'unknown')
        epoch = row.get('Epoch', 'DNF')
        f1 = row.get('f1_macro', 0)
        rank_str = f"**{i}**"
        net_str = f"**{network}**"
        f1_str = f"**{f1:.10f}**"
        ranked_table += f"| {rank_str} | {net_str} | {epoch} | {f1_str} | |\n"

    full_table = "\n\n### Full Scores Results\n\n"
    full_table += "| Rank | Network | Epoch (MSE<0.02) | F1 Score | MCC | Accuracy | Precision | Recall |\n"
    full_table += "| --- | --- | --- | --- | --- | --- | --- | --- |\n"

    for i, row in enumerate(sorted_by_f1, 1):
        network = row.get('network', 'unknown')
        epoch = row.get('Epoch', 'DNF')
        f1 = row.get('f1_macro', '—')
        mcc = row.get('matthews', '—')
        accuracy = row.get('accuracy', '—')
        precision = row.get('precision_macro', '—')
        recall = row.get('recall_macro', '—')

        rank_str = f"**{i}**"
        net_str = f"**{network}**"
        f1_str = f"**{f1}**" if f1 != '—' else f1

        full_table += f"| {rank_str} | {net_str} | {epoch} | {f1_str} | {mcc} | {accuracy} | {precision} | {recall} |\n"

    time_table = generateTimeAdjustedTable(sorted_by_f1)

    markdown_content = ranked_table + full_table + time_table

    mdpath = os.path.join('final', 'results.md')
    with open(mdpath, 'w') as f:
        f.write(markdown_content)
    print(f'Results markdown written to {mdpath}')

def generateTimeAdjustedTable(data):
    time_table = "\n\n### Training Time Results\n\n"
    time_table += "Adding in a value so that we can compare normalized training time to get an equivalent 0.02 MSE:\n\n"
    time_table += "$$ \\text{Time-Adjusted} = \\text{Time} × (\\text{Epoch}_{\\text{converged}} / 100) $$\n\n"
    time_table += "Special rules:\n\n"
    time_table += "* If **Epoch\\_converged > 100**, multiplier = **1.00**\n"
    time_table += "* If **DNF**, then Time-Adjusted = **N/A**\n\n"
    time_table += "| Rank | Network | Epoch | Time (s) | Time-Adjusted | Your Notes |\n"
    time_table += "| --- | --- | --- | --- | --- | --- |\n"

    for i, row in enumerate(data, 1):
        network = row.get('network', 'unknown')
        epoch_str = row.get('Epoch', 'DNF')
        time_val = float(row.get('Time', 0))

        rank_str = f"**{i}**"
        net_str = f"**{network}**"

        try:
            epoch_num = float(epoch_str)
            if epoch_num > 100:
                time_adj = time_val
            else:
                time_adj = time_val * (epoch_num / 100)
            time_adj_str = f"**{time_adj:.5f}**"
        except (ValueError, TypeError):
            time_adj_str = "**N/A**"

        time_table += f"| {rank_str} | {net_str} | {epoch_str} | {time_val:.5f} | {time_adj_str} | |\n"

    return time_table

if __name__ == '__main__':
    exportdir = 'export'
    outputfile = 'final/combined_scores.csv'
    combineScores(exportdir, outputfile)