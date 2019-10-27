# save-the-babies
General utilities for reading fetal cardiotocography data from https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

## Read Waveform files from Physionet.org

Get the wavefrm data from the physionet archive:

`wget --recursive --no-parent https://physionet.org/physiobank/database/ctu-uhb-ctgdb/`

Read them into python:

```python
python read_physionet.py --files physiobank/database/ctu-uhb-ctgdb/ --title 1001

```


![](assets/ctu_uhb_1.png)
