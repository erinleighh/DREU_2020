import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


data = pd.read_csv('resultsTable.csv')
objects = data['Obj ID'].drop_duplicates()
resultsTable = pd.DataFrame()

for obj in objects:
    print("\nReading in " + obj)
    observations = data['Classification'].loc[data['Obj ID'] == obj]
    if any(observations == 'EB'):
        print('At least one EB found!')
        objTable = data.loc[data['Obj ID'] == obj]
        classification = objTable['Classification'].tolist()
        files = objTable['Filename'].copy()
        
        resultsTable = resultsTable.append(objTable, ignore_index=True)
        
        i = 0

        # Plot all observations on different light curves
        for file in files:
            plt.figure(figsize=(10, 5))
            fitsTable = fits.open(file, memmap=True)
            curveTable = Table(fitsTable[1].data).to_pandas()
            curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
            fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
            curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
            curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)
            
            # Determine EB/non-EB and colorize accordingly
            if classification[i] == 'EB':
                color = 'tab:purple'
            else:
                color = 'tab:gray'
            
            if i < 10:
                figName = obj + '_' + '0' + str(i) + '.png'
            else: 
                figName = obj + '_' + str(i) + '.png'
                
            plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color=color, s=.2)
            
            plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
            plt.ylabel('Relative Flux')
            plt.title('Light Curve for ' + obj + '\n' + 'Observation ' + str(i) + '\n' + file)

            if classification[i] == 'EB':
                path = 'EB'
            else:
                path = 'und'

            plt.savefig(os.path.join(path, figName), orientation='landscape')
            plt.close()
            
            i += 1
        
    else:
        print('No EB found.')
    print(obj + " complete.")

try:
    resultsTable = resultsTable.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
except:
    resultsTable = resultsTable.drop(['Unnamed: 0'], axis=1)
resultsTable.to_csv('EBresults.csv')