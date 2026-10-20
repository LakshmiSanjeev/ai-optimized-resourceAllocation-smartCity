import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'tempResults'))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'LTEResults.csv')

DIRECTORIES = ['PF_20', 'PF_40', 'PF_80', 'PF_120']

def loadCSV(filepath):
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath, sep='\t', index_col=False)
    df.columns = df.columns.str.strip().str.lower()
    return df

def parse(path, num_ues):
    
    sinr_file = 'DlRsrpSinrStats.txt'
    mac_file = 'DlMacStats.txt'
    rlc_file = 'DlRlcStats.txt'
    mcs_col = 'mcstb1'
    sinr_col = 'sinr'
        
    sinrData = loadCSV(os.path.join(path, sinr_file))
    macData = loadCSV(os.path.join(path, mac_file))
    rlcData = loadCSV(os.path.join(path, rlc_file))
    
    if sinrData.empty or macData.empty or rlcData.empty:
        print(f"  Warning: Missing files in {path}. Skipping.")
        return pd.DataFrame()

    sinrData = sinrData.rename(columns={'% time': 'time'})
    
    if sinr_col not in sinrData.columns:
        raise KeyError(f"Column '{sinr_col}' not found in {sinr_file}. Available columns: {sinrData.columns.tolist()}")

    sinrData['SINR_dB'] = 10 * np.log10(sinrData[sinr_col])
    sinrData['SINR_dB'] = sinrData['SINR_dB'].clip(lower=-30.0, upper=40.0)
    sinrData = sinrData[['time', 'imsi', 'SINR_dB']].sort_values('time')

    macData = macData.rename(columns={'% time': 'time', mcs_col: 'MCS'})
    macData = macData[['time', 'imsi', 'MCS']].sort_values('time') 

    rlcData = rlcData.rename(columns={'% start': 'time'})
    duration = rlcData['end'] - rlcData['time']
    rlcData['Throughput'] = (rlcData['rxbytes'] * 8) / duration / 1e6
    rlcData = rlcData.rename(columns={'delay': 'Delay', 'txbytes': 'Queue'})
    rlcData = rlcData[['time', 'imsi', 'Throughput', 'Queue', 'Delay']].sort_values('time')

    phyMac = pd.merge_asof(sinrData, macData, on='time', by='imsi', direction='nearest')
    traceData = pd.merge_asof(phyMac, rlcData, on='time', by='imsi', direction='backward')
    traceData = traceData.dropna()

    traceData['bin'] = (traceData['time'] / 0.05).round() * 0.05
    traceData = traceData.sort_values('time')
    traceData = traceData.groupby(['imsi', 'bin']).first().reset_index()
    traceData = traceData.drop(columns=['bin'])
    
    traceData['Num_UEs'] = num_ues
    
    traceData['AvgThroughput'] = traceData.groupby('imsi')['Throughput'].transform(lambda x: x.expanding().mean())
    
    sinr_norm = (traceData['SINR_dB'] - traceData['SINR_dB'].min()) / (traceData['SINR_dB'].max() - traceData['SINR_dB'].min())
    queue_norm = traceData['Queue'] / traceData['Queue'].max()

    traceData['Label'] = (
        (traceData['Throughput'] / (traceData['AvgThroughput'] + 1e-6))
        + 0.3 * sinr_norm
        + 0.2 * queue_norm
        - 0.3 * traceData['Delay']
    )

    traceData['Queue'] = traceData['Queue'].astype(int)
    traceData = traceData.rename(columns={'time': 'Time', 'imsi': 'IMSI', 'Num_UEs':'nUEs'})
    final = ['Time', 'IMSI', 'nUEs', 'SINR_dB', 'MCS', 'Queue', 'Delay', 'Throughput', 'AvgThroughput', 'Label']
    
    return traceData[final]

def processResults():
    data = []

    for folder in DIRECTORIES:
        print(f"Processing directory: {folder}...")
        path = os.path.join(BASE_DIR, folder)
        num_ues = int(folder.split('_')[1])
        
        DLData = parse(path, num_ues)
        
        if not DLData.empty: data.append(DLData)

    if data:
        finalDataset = pd.concat(data, ignore_index=True)
        finalDataset = finalDataset.sort_values(by=['Time', 'IMSI'])
        finalDataset.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccess! Saved {len(finalDataset)} time-aligned rows to:\n{OUTPUT_FILE}")
    else:
        print("\nFailed to process data. Check directory paths and file contents.")

if __name__ == "__main__":
    processResults()