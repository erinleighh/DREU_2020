import glob
import os
import pandas as pd
import numpy as np
from scipy import stats
import math
import exoplanet as xo
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def findsd(relFlux, medianFlux):
    # Calculate z-score of all points, find outliers below the flux midpoint.
    z = stats.median_abs_deviation(relFlux)
    # try:
    #     maximumSD = np.max(z)
    # except:
    #     maximumSD = np.NaN

    return z


def autocorrelationfn(time, relFlux, relFluxErr):
    acf = xo.autocorr_estimator(time.values, relFlux.values, yerr=relFluxErr.values,
                                min_period=0.1, max_period=27, max_peaks=10)

    period = acf['autocorr'][0]
    power = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(power)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    maxPower = np.max(acfLocalMaxima).values

    bestPeriod = period[np.where(power == maxPower)[0]][0]
    peaks = acf['peaks'][0]['period']

    k = range(2, 20)

    window = peaks / np.abs(np.nanmedian(np.diff(time))) / k
    window = window.astype(int)

    return period, power, bestPeriod, maxPower, window, k


numTab = 11  # We have 11 fits files to read from.
lightCurves = [0] * numTab  # Store the light curves for all the tables.
path = "data"  # Hack for dealing with OS forward/back slash conflicts.
EBs = []  # Store the objects classified as eclipsing binaries

for file in glob.glob(os.path.join(path, "*.fits")):
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    print("\nReading in " + objName)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
    fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
    curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
    curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

    # Calculate original maximum SD below the median
    originalMaxSD = findsd(curveData['REL_FLUX'], fluxMed)

    # Autocorrelation Function using exoplanet.
    print("Running ACF.")
    acfPeriod, acfPower, acfBestPeriod, acfMaxPower, windows, k = autocorrelationfn(curveData['TIME'],
                                                                                    curveData['REL_FLUX'],
                                                                                    curveData['REL_FLUX_ERR'])

    # Smoothing
    print("Performing smoothing on " + objName)

    mad = []
    for s_window in windows:
        smoothedFlux = curveData['REL_FLUX'].rolling(s_window, center=True).median()
        SOK = np.isfinite(smoothedFlux)
        newFlux = curveData['REL_FLUX'][SOK] - smoothedFlux[SOK]
        curveData['REL_FLUX'] = newFlux.copy()
        curveData = curveData.dropna(subset=['TIME']).dropna(subset=['REL_FLUX']).dropna(subset=['REL_FLUX_ERR']).copy()
        fluxMed = np.nanmedian(curveData['REL_FLUX'])

        maxSD = findsd(curveData['REL_FLUX'], fluxMed)
        mad.append(maxSD)

    # Graphing
    print("Generating plots for " + objName)

    plt.figure(figsize=(12, 9))
    figName = objName + '.png'

    # Graphing MAD vs window size.
    plt.subplot(211)
    plt.scatter(windows, mad, color='tab:purple', s=5)
    plt.axhline(originalMaxSD)
    plt.xlabel('Smoothing Window Size')
    plt.ylabel('Max. Deviation')
    plt.title('Max. Deviation vs Window Size for ' + objName)

    # Graphing MAD vs k.
    plt.subplot(212)
    plt.scatter(k, mad, color='tab:purple', s=5)
    plt.axhline(originalMaxSD)
    plt.xlabel('Smoothing Window Kernel (k)')
    plt.ylabel('Max. Deviation')
    plt.title('Max. Deviation vs Window Kernel for ' + objName)

    plt.suptitle('Magic Number Plots for ' + figName)
    plt.savefig(os.path.join('mnPlots', figName), orientation='landscape')
    plt.close()
