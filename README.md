# Roz

![logo](images/Roz.png)

_"Well, isn't that nice? But guess what? You didn't turn in your paperwork last night."_

## Description:

Filing Cabinet and Quality Control for LDT Instrument Calibration Frames

See https://jumar.lowell.edu/confluence/display/LIG/Quality+Assurance+of+Instrument+Data for concept flow chart.

### Initial Functionality:
- When fed directory information by an automatic script, find and analyze LMI flat field frames.
- Add analysis information to a database (somewhere).
- Create a dynamic page to replace https://jumar.lowell.edu/confluence/display/LDTOI/LMI+Filter+Characterization

### Future Functionality:
- Analyze bias frames from LMI and DeVeny to find trends with mount temperature and other conditions.
- Ensure some level of repeatability.
- Automated monitoring of top-ring lamp spectra (DeVeny DV1) with respect to May 2021 lamp replacements.
- Track LMI DIQ on a nightly basis.

## Requirements:
- Python 3.8+
- Astropy 4.0+
- CCDPROC 2.1+
- Numpy
- Matplotlib
- Atlassian-Python-Api
- BeautifulSoup
- tqdm
- Importlib-Resources
- LowellObservatory/ligmos
