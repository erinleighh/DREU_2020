from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import altair as alt
import glob
import os
import pandas as pd

numTab = 11  # We have 11 fits files to read from.
lightCurves = [0] * numTab  # Store the light curves for all the tables.
path = "start_data"  # Hack for dealing with OS forward/back slash conflicts.
i = 0

for file in glob.glob(os.path.join(path, "*.fits")):
    print("\nReading in " + str(file))
    fitsTable = fits.open(file, memmap=True)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable[curveTable['QUALITY'] == 0]
    curveData['PDCSAP_FLUX']=curveData['PDCSAP_FLUX']/curveData['PDCSAP_FLUX'].median()

    # matplotlib graphing
    plt.scatter(curveData['TIME'], curveData['PDCSAP_FLUX'], color='tab:purple', s=.5)
    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
    plt.ylabel('Relative Flux')
    plt.title('Light Curve')

    i += 1
    figName = 'LightCurve' + str(i) + '.png'
    plt.savefig(os.path.join('curves', figName), orientation='landscape')
    plt.clf()

    # Altair graphing that doesn't work yet
    # chart = alt.Chart(curve).mark_circle(size=1).encode(
    #     x='TIME',
    #     y='PDCSAP_FLUX',
    #     tooltip=['TIME', 'PDCSAP_FLUX']
    # ).interactive()
    #
    # i += 1
    # saveFile = os.path.join(str(i), 'chart.html')
    # chart.save(saveFile)
