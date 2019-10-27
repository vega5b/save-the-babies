# save-the-babies
General utilities for reading fetal cardiotocography data from https://physionet.org/content/ctu-uhb-ctgdb/1.0.0/

## Read Waveform files from Physionet.org

Get the wavefrm data from the physionet archive:

`wget --recursive --no-parent https://physionet.org/physiobank/database/ctu-uhb-ctgdb/`

Read them into python:

`python read_physionet.py --files physiobank/database/ctu-uhb-ctgdb/ --title 1001`


![](assets/ctu_uhb_1.png)

## Convert pdf's to individual pngs

`brew install ghostscript`
 
`gs -dNOPAUSE -sDEVICE=png16m -r256 -sOutputFile=myData/page%03d.png myData.pdf`

(you may need to type `quit` once the GS prompt is returned)
