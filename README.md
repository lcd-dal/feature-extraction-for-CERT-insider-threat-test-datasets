# Feature extraction for CERT insider threat test dataset
This is a script for extracting features (csv format) from the [CERT insider threat test dataset](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099), versions 4.1 to 6.2. For more details, please see this paper: [Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning](https://ieeexplore.ieee.org/document/8962316).

## Run the script
- Require python3, numpy, pandas, joblib. The script is written and tested on Linux only.
- By default the script extracts week, day, session, and sub-session data (as in the paper).
- To run the script, place it in a folder of a CERT dataset (e.g. r4.2, decompressed from r4.2.tar.bz2 downloaded [here](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1)), then run `python3 featureExtraction.py`
- To change number of cores used in parallelization (default 8), use `pyhon3 featureExtraction.py numberOfCores`, e.g `python3 featureExtraction.py 16`.

## Extracted Data
Extracted data is stored in ExtractedData subfolder.

Note that in the extracted data, `insider` is the label indicating the insider threat scenario (0 is normal). Some extracted features (subs_ind, starttime, endtime, sessionid, user, day, week) are for information and may or may not be used in training machine learning approaches.

Pre-extracted data from CERT insidere threat test dataset r5.2 can be found in this [Onedrive folder](https://web.cs.dal.ca/~lcd/data/CERTr5.2/).


## Citation
D. C. Le, N. Zincir-Heywood and M. I. Heywood, "Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning," in IEEE Transactions on Network and Service Management, vol. 17, no. 1, pp. 30-44, March 2020, doi: 10.1109/TNSM.2020.2967721.
