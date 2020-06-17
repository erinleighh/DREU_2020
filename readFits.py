from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import altair as alt
import glob
import os

numTab = 11  # We have 11 fits files to read from.
lightCurves = [0] * numTab  # Store the light curves for all the tables.
path = "start_data"  # Hack for dealing with OS forward/back slash conflicts.
i = 0

for file in glob.glob(os.path.join(path, "*.fits")):
    print("\nReading in...")
    fitsTable = fits.open(file, memmap=True)
    fitsTable.info()
    curveData = Table(fitsTable[1].data)

    # matplotlib graphing
    plt.scatter(curveData['TIME'], curveData['SAP_FLUX'], color='tab:purple', s=.5)
    plt.xlabel('Time (days)')
    plt.ylabel('Flux')
    plt.title('Light Curve')

    i += 1
    figName = 'LightCurve' + str(i) + '.png'
    plt.savefig(os.path.join('curves', figName), orientation='landscape')
    plt.clf()

    # Altair graphing that doesn't work yet
    # chart = alt.Chart(curve).mark_circle(size=1).encode(
    #     x='TIME',
    #     y='SAP_FLUX',
    #     tooltip=['TIME', 'SAP_FLUX']
    # ).interactive()
    #
    # i += 1
    # saveFile = os.path.join(str(i), 'chart.html')
    # chart.save(saveFile)
