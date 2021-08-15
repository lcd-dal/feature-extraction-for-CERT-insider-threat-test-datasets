# Feature extraction for CERT insider threat test dataset
This is a script for extracting features (csv format) from the [CERT insider threat test dataset](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099) [[1]](#1), [[2]](#2), versions 4.1 to 6.2. For more details, please see this paper: [Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning](https://ieeexplore.ieee.org/document/8962316).

<a id="1">[1]</a> 
Lindauer, Brian (2020): Insider Threat Test Dataset. Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/12841247.v1 

<a id="2">[2]</a> 
J. Glasser and B. Lindauer, "Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data," 2013 IEEE Security and Privacy Workshops, San Francisco, CA, 2013, pp. 98-104, doi: 10.1109/SPW.2013.37.

## Run feature_etraction script
- Require python3, numpy, pandas, joblib. The script is written and tested in Linux only.
- By default the script extracts week, day, session, and sub-session data (as in the paper).
- To run the script, place it in a folder of a CERT dataset (e.g. r4.2, decompressed from r4.2.tar.bz2 downloaded [here](https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247/1)), then run `python3 feature_etraction.py`
- To change number of cores used in parallelization (default 8), use `python3 feature_etraction.py numberOfCores`, e.g `python3 feature_etraction.py 16`.

## Extracted Data
Extracted data is stored in ExtractedData subfolder.

Note that in the extracted data, `insider` is the label indicating the insider threat scenario (0 is normal). Some extracted features (subs_ind, starttime, endtime, sessionid, user, day, week) are for information and may or may not be used in training machine learning approaches.

Pre-extracted data from CERT insider threat test dataset r5.2 (gzipped) can be found in [here](https://web.cs.dal.ca/~lcd/data/CERTr5.2/).

## Data representations
From the extracted data, `temporal_data_representation.py` can be used to generate different data representations, as presented in this paper: [Anomaly Detection for Insider Threats Using Unsupervised Ensembles](https://ieeexplore.ieee.org/document/9399116). 

`python3 temporal_data_representation.py --help`

## Sample classification and anomaly detection results
Sample code is provided in:

- `sample_classification.py` for classification (as in [Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning](https://ieeexplore.ieee.org/document/8962316)).
- `sample_anomaly_detection.py` for anomaly detection (as in [Anomaly Detection for Insider Threats Using Unsupervised Ensembles](https://ieeexplore.ieee.org/document/9399116)).

## Citation
If you use the source code, or the extracted datasets, please cite the following paper:

`D. C. Le, N. Zincir-Heywood and M. I. Heywood, "Analyzing Data Granularity Levels for Insider Threat Detection Using Machine Learning," in IEEE Transactions on Network and Service Management, vol. 17, no. 1, pp. 30-44, March 2020, doi: 10.1109/TNSM.2020.2967721.`

Data representations and anomaly detection:

`D. C. Le, N. Zincir-Heywood, "Anomaly Detection for Insider Threats Using Unsupervised Ensembles," in IEEE Transactions on Network and Service Management, vol. 18, no. 2, pp. 1152–1164. June 2021, doi:http://doi.org/10.1109/TNSM.2021.3071928.`
