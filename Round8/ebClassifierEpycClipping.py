import glob
import os
import pandas as pd
import numpy as np
from scipy import stats
import exoplanet as xo
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle, BoxLeastSquares
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time as timer

start = timer.time()


def classification(blsMP, sd, factor):
    resClassification = 'notEB'

    # Line from SD 3, BLS 1000 to SD 15, BLS 100
    # minblsMP = (-75 * sd) + min((20 * factor * sd), (75 * sd)) + 1225 - min(factor * 250, 1025)
    minblsMP = startMP - dropMP * factor

    # Identify significant eclipses
    if blsMP >= minblsMP or sd >= 10 - 2 * factor:
        resClassification = 'EB'
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')

    return resClassification


def findsd(relFlux, time):
    # Calculate z-score of all points, find outliers below the flux midpoint.
    z = stats.zscore(relFlux)
    potentialEclipses = z[np.where(relFlux < 1)[0]]
    peTimes = time.iloc[np.where(relFlux < 1)[0]]

    maxSDRangeIndex = range(max(np.argmin(potentialEclipses) - 2, 0),
                            min(np.argmin(potentialEclipses) + 3, potentialEclipses.size))
    sdRange = np.ceil(potentialEclipses[maxSDRangeIndex]).astype(int)

    maximumSD = np.min(sdRange) * -1
    avgMaximumSD = np.average(sdRange).round(1) * -1
    # To ensure that data points near the max SD are indeed significant

    return maximumSD, avgMaximumSD, potentialEclipses[maxSDRangeIndex] * -1, peTimes.iloc[maxSDRangeIndex], \
           relFlux.iloc[maxSDRangeIndex]


def autocorrelationfn(time, relFlux, relFluxErr):
    acf = xo.autocorr_estimator(time.values, relFlux.values, yerr=relFluxErr.values,
                                min_period=0.05, max_period=27, max_peaks=10)

    period = acf['autocorr'][0]
    power = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(power)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    maxPower = np.max(acfLocalMaxima).values

    bestPeriod = period[np.where(power == maxPower)[0]][0]
    peaks = acf['peaks'][0]['period']

    if len(acf['peaks']) > 0:
        window = int(peaks / np.abs(np.nanmedian(np.diff(time))) / k)
    else:
        window = 128

    return period, power, bestPeriod, maxPower, window


def boxleastsquares(time, relFlux, relFluxErr, acfBP):
    model = BoxLeastSquares(time.values, relFlux.values, dy=relFluxErr.values)
    duration = [20 / 1440, 40 / 1440, 80 / 1440, .1]
    periodogram = model.power(period=[.5 * acfBP, acfBP, 2 * acfBP], duration=duration,
                              objective='snr')
    period = periodogram.period
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    return period, power, bestPeriod, maxPower


def addtotable(table, oID, blsMP, acfMP, mSD, amSD, smooth, flg, c, rSD, f):
    table = table.append(
        {'Obj ID': oID, 'BLS Max Pow': blsMP, 'ACF Max Pow': acfMP, 'Max SD': mSD,
         'AvgMax SD': amSD, 'Times Smoothed': smooth, 'Flag': flg, 'Classification': c,
         'SD Range': rSD, 'Filename': f}, ignore_index=True)
    return table


def makegraph(xaxis, yaxis, xlabels, ylabels, lbl, color, marker=None, size=None, style=None, ax=None):
    if ax is None:
        ax = plt.gca()
    if style is None:
        ax.scatter(xaxis, yaxis, color=color, marker=marker, s=size)
    else:
        ax.plot(xaxis, yaxis, color=color)

    plt.xlabel(xlabels)
    plt.ylabel(ylabels)
    plt.title(lbl)
    return ax


lightCurves = []  # Initialize the array holding light curves
path = "data_all"  # Folder containing fits files
EBs = []  # Store the objects classified as eclipsing binaries

k = 5
startMP = 1500
dropMP = 650
data = pd.read_csv('/data/epyc/users/jrad/TESS_CVZ/001_026_3S.csv')
files = data['file']

lcTable = pd.DataFrame(
    columns=['Obj ID', 'BLS Max Pow', 'ACF Max Pow', 'Max SD', 'AvgMax SD', 'Times Smoothed', 'Flag', 'Classification',
             'SD Range', 'Filename'])

for file in files:
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    print("\nReading in " + objName + " Filename: " + file)
    try:
        curveTable = Table(fitsTable[1].data).to_pandas()
    except:
        print('*************** ERROR ***************')
        f = open('errors.txt', 'a')
        f.write(file + '\n')
        f.close()
    else:
        curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
        curveData = curveTable.filter(['TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'])

        idx = np.where((curveData['TIME'][1:]-curveData['TIME'][:-1]).isnull())[0]
        idxL = idx[np.where(idx[1:]-idx[:-1] > 1)]
        idxR = idx[np.where(idx[1:]-idx[:-1] > 1)[0]+1]

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
        curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

        originalFlux = curveData['REL_FLUX'].copy()
        originalTime = curveData['TIME'].copy()

        classif = 'notEB'
        flag = ''

        # Classify based on outliers
        i = 0
        while classif == 'notEB' and i < 3:  # Potential to be an EB.
            maxSD, avgMaxSD, rangeSD, fluxRange, timeRange = findsd(curveData['REL_FLUX'], curveData['TIME'])

            if avgMaxSD < 2.8 or min(rangeSD) < 1:
                # Pre-classification
                # Max SD of 0 through 2 highly unlikely to be eclipse.
                # If one of the data points is less than one, likely to be error.
                
                # Add to table
                print("Adding to table.")
                try:
                    lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag, classif, rangeSD, file)
                except:
                    pass
                break
            else:
                # Run ACF and BLS functions for classification

                try:
                    # Autocorrelation Function
                    print("Generating ACF periodogram.")
                    acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'])

                    # Box Least Squares
                    print("Generating BLS periodogram.")
                    BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'], curveData['REL_FLUX'], curveData['REL_FLUX_ERR'], acfBestPeriod)
                except:
                    break

                # Additional pre-classification
                if (i == 0 and BLSmaxPower < 100 and avgMaxSD < 7) or \
                        (acfMaxPower < 0.05 and avgMaxSD < 4) or BLSmaxPower < 60:
                    # No need to smooth attempt further, very unlikely to be obvious EBs.

                    # Add to table
                    print("Adding to table.")
                    lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag, classif, rangeSD, file)
                    break

                # Run classification
                classif = classification(BLSmaxPower, avgMaxSD, i)

                if classif == 'notEB':
                    # Perform Smoothing
                    print("Performing smoothing on " + objName)
                    smoothedFlux = curveData['REL_FLUX'].rolling(s_window, center=True).median()

                    SOK = np.isfinite(smoothedFlux)

                    newFlux = curveData['REL_FLUX'][SOK] - smoothedFlux[SOK]

                    curveData['REL_FLUX'] = newFlux.copy()

                    curveData = curveData.dropna(subset=['TIME']).dropna(subset=['REL_FLUX']).dropna(
                        subset=['REL_FLUX_ERR']).copy()

                    fluxMed = np.nanmedian(curveData['REL_FLUX'])
                else:
                    EBs.append(objName)  # Add to printout of EBs

                # Add to table
                print("Adding to table.")
                lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag, classif, rangeSD, file)

                i += 1

    print(objName + " complete.")

print('\nClassification complete.\n')

# Print table to file
print("\nPrint curve table to file.\n")
lcTable.to_csv('curvesTable.csv')

print("\nProcess complete.\n")

EBs = list(dict.fromkeys(EBs))
print('EBs found: ' + str(len(EBs)))

end = timer.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
