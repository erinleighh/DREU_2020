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

        curveData = pd.DataFrame()
        # Plot all observations on the same light curve
        plt.figure(figsize=(20, 6))
        for file in files:
            fitsTable = fits.open(file, memmap=True)
            curveTable = Table(fitsTable[1].data).to_pandas()
            curveDataNew = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
            curveData = curveData.append(curveDataNew, ignore_index=True)
            
            # Determine EB/non-EB and colorize accordingly
            if classification[i] == 'EB':
                color = 'tab:purple'
            else:
                color = 'tab:gray'
            
            #plt.axvspan(np.min(curveDataNew['TIME'].dropna()), np.max(curveDataNew['TIME'].dropna()), ymin=np.min(curveDataNew['PDCSAP_FLUX'].dropna()), ymax=np.max(curveDataNew['PDCSAP_FLUX'].dropna()), alpha=0.5, color=color)
            
            
        curveData = curveTable.filter(['TIME', 'PDCSAP_FLUX'])

        idx = np.where((curveData['TIME'][1:]-curveData['TIME'][:-1]).isnull())[0]
        idxL = idx[np.where(idx[1:]-idx[:-1] > 1)]
        idxR = idx[np.where(idx[1:]-idx[:-1] > 1)[0]+1]

        curveData.to_csv('time1.csv')

        for badDataPoint in idxL:
            # Set data points to the right to null
            r = range(badDataPoint + 1, badDataPoint + 1001)

            try:
                curveData.loc[r, 'PDCSAP_FLUX'] = None
                curveData.loc[r, 'TIME'] = None
            except:
                pass

        for badDataPoint in idxR:
            # Set data points to the left to null
            l = range(badDataPoint - 1000, badDataPoint)

            try:
                curveData.loc[l, 'PDCSAP_FLUX'] = None
                curveData.loc[l, 'TIME'] = None
            except:
                pass

        curveData = curveData.dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
        fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
        curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)

        figName = obj + '.png'
        plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:cyan', s=.2)

        i += 1
            
        plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
        plt.ylabel('Relative Flux')
        plt.title('Light Curve for ' + obj)
        plt.savefig(os.path.join('EBjoined', figName), orientation='landscape')
        plt.close()
        
    else:
        print('No EB found.')
    print(obj + " complete.")