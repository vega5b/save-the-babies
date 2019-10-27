#wget --recursive --no-parent https://physionet.org/physiobank/database/ctu-uhb-ctgdb/

import wfdb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files',help='path to physionet files, e.g. physiobank/database/ctu-uhb-ctgdb/')  
parser.add_argument('--title', help='the numeric data file prefix, e.g. 1001')     
args = parser.parse_args()

rec=args.files+args.title
record = wfdb.rdrecord(rec)
wfdb.plot_wfdb(record=record, title=args.title)
signals, fields = wfdb.rdsamp(rec)

#annotation = wfdb.rdann(rec, 'atr')
# signals are heartbeat and contraction
