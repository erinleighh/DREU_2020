import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


lightCurves = []  # Initialize the array holding light curves
path = "data3"  # Folder containing fits files

for file in glob.glob(os.path.join(path, "*.fits")):
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    print("\nReading in " + objName)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
    fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
    curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
    curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

    # Plot with various axes and scales.
    plt.figure(figsize=(16, 12))

    # matplotlib graphing
    print("Generating light curve.")
    figName = objName + '.png'
    plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
    plt.ylabel('Relative Flux')
    plt.title('Light Curve for ' + objName)
    plt.savefig(os.path.join('curves', figName), orientation='landscape')
    plt.close()
    print(objName + " complete.")
