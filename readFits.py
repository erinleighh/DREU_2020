import glob
import os

import altair as alt
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

numTab = 11  # We have 11 fits files to read from.
lightCurves = [0] * numTab  # Store the light curves for all the tables.
path = "start_data"  # Hack for dealing with OS forward/back slash conflicts.
i = 0

for file in glob.glob(os.path.join(path, "*.fits")):
    print("\nReading in " + str(file))
    fitsTable = fits.open(file, memmap=True)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable.loc[curveTable['QUALITY'] == 0].copy()
    fluxMed = curveData['PDCSAP_FLUX'].median()
    curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)

    # matplotlib graphing
    plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
    plt.ylabel('Relative Flux')
    plt.title('Light Curve')

    i += 1
    figName = 'LightCurve' + str(i) + '.png'
    plt.savefig(os.path.join('curves', figName), orientation='landscape')
    plt.clf()

    # Altair interactive graphing
    chart = alt.Chart(curveData).mark_circle(size=1).encode(
        alt.X('TIME', axis=alt.Axis(title='BJD - 2457000 (days)'), scale=alt.Scale(zero=False)),
        alt.Y('REL_FLUX', axis=alt.Axis(title='Relative Flux'), scale=alt.Scale(zero=False)),
        tooltip=['TIME', 'REL_FLUX']
    ).properties(title='Light Curve').interactive()

    saveFile = os.path.join('curves', str(i) + 'chart.html')
    chart.save(saveFile)
