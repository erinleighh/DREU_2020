import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


data = pd.read_csv('/astro/users/elhoward/DREU_2020/Round2/curvesTable.csv')
objects = data['Obj ID'].drop_duplicates()

for obj in objects:
    print("\nReading in " + obj)
    if any(data['Classification'].loc[data['Obj ID'] == obj] == 'EB'):
        print('At least one EB found!')
        objTable = data.loc[data['Obj ID'] == obj]
        classification = objTable['Classification'].tolist()
        files = objTable['Filename'].copy()
        i = 0

        # Plot all observations on the same light curve
        plt.figure(figsize=(20, 6))
        for file in files:
            fitsTable = fits.open(file, memmap=True)
            curveTable = Table(fitsTable[1].data).to_pandas()
            curveData = curveTable.loc[curveTable['QUALITY'] == 0].groupby(['TIME'], as_index=False).last()

            idx = curveData.index[curveData['PDCSAP_FLUX'].apply(np.isnan)]

            for badDataPoint in idx:
                a = range(badDataPoint - 500, badDataPoint)
                b = range(badDataPoint + 1, badDataPoint + 501)

                try:
                    curveData.loc[a, 'PDCSAP_FLUX'] = None
                except:
                    pass
                try:
                    curveData.loc[b, 'PDCSAP_FLUX'] = None
                except:
                    pass

            curveData = curveData.dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
            fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
            curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
            curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)
            
            # Determine EB/non-EB and colorize accordingly
            if classification[i] == 'EB':
                color = 'tab:purple'
            else:
                color = 'tab:gray'
                
            figName = obj + '.png'
            plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color=color, s=.2)
            
            i += 1
            
        plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
        plt.ylabel('Relative Flux')
        plt.title('Light Curve for ' + obj)
        plt.savefig(os.path.join('EBjoined', figName), orientation='landscape')
        plt.close()
        
    else:
        print('No EB found.')
    print(obj + " complete.")