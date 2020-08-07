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
knownData = 0  # Boolean. Turns on/off diagnostic graphing. Prints confirmed light curves only.


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

    maximumSD = abs(np.min(sdRange))
    avgMaximumSD = np.average(sdRange).round(0).astype(int) * -1
    # To ensure that data points near the max SD are indeed significant

    return maximumSD, avgMaximumSD, potentialEclipses[maxSDRangeIndex] * -1, peTimes.iloc[maxSDRangeIndex], \
           relFlux.iloc[maxSDRangeIndex]


def lombscargle(time, relFlux):
    LS = LombScargle(time, relFlux)
    frequency, power = LS.autopower(minimum_frequency=1 / 27, maximum_frequency=1 / .05)
    bestPeriod = 1 / frequency[np.argmax(power)]
    maxPower = np.max(power)
    period = 1 / frequency

    return period, power, bestPeriod, maxPower


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

if knownData == 1:
    confirmedEBs = [line.rstrip('\n') for line in open('curves.txt')]  # Confirmed by eye EBs from data

k = 5
startMP = 1500
dropMP = 650
data = pd.read_csv('/data/epyc/users/jrad/TESS_CVZ/001_025_10S.csv')
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
        fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
        curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
        curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

        originalFlux = curveData['REL_FLUX'].copy()
        originalTime = curveData['TIME'].copy()

        classif = 'notEB'

        if knownData == 1:
            # Check if marked as confirmed EB
            if objName in confirmedEBs:
                flag = 'EB'
            else:
                flag = 'notEB'
        else:
            flag = ''

        # Classify based on outliers
        i = 0
        while classif == 'notEB' and i < 3:  # Potential to be an EB.
            maxSD, avgMaxSD, rangeSD, fluxRange, timeRange = findsd(curveData['REL_FLUX'], curveData['TIME'])

            if avgMaxSD < 3:
                # Pre-classification
                # Max SD of 0 through 2 highly unlikely to be eclipse.
                if knownData == 1 and flag == 'EB':
                    title = 'Misclassified: ' + objName + "\n" + str(i) + "x smoothing" + "\n" + file

                    # Add to table
                    print("Adding to table.")
                    lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag, classif,
                                         rangeSD, file)

                    figName = 'missedEB_' + objName + '.png'

                    if i == 0:
                        # Graph the light curve vs. smoothed light curve
                        print("Generating misclassified chart.")
                        plt.figure(figsize=(9, 6))
                        makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux',
                                  'Light Curve for ' + objName, 'tab:purple', '.', .2)
                        plt.suptitle(title)
                        plt.savefig(os.path.join('misclassified', figName), orientation='landscape')
                        plt.close()

                    else:
                        plt.figure(figsize=(9, 9))
                        # Graph the light curve vs. smoothed light curve
                        print("Generating misclassified chart.")
                        plt.subplot(211)
                        makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux',
                                  'Light Curve for ' + objName, 'tab:purple', '.', .2)

                        plt.subplot(212)
                        makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                                  'Smoothed ' + str(i) + 'x light curve for ' + objName, 'tab:purple', '.', .2)

                        plt.suptitle(title)
                        plt.savefig(os.path.join('misclassified', figName), orientation='landscape')
                        plt.close()
                break
            else:
                # Run ACF and BLS functions for classification

                try:
                    # Autocorrelation Function
                    print("Generating ACF periodogram.")
                    acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'],
                                                                                                  curveData['REL_FLUX'],
                                                                                                  curveData[
                                                                                                      'REL_FLUX_ERR'])

                    # Box Least Squares
                    print("Generating BLS periodogram.")
                    BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'],
                                                                                      curveData['REL_FLUX'],
                                                                                      curveData['REL_FLUX_ERR'],
                                                                                      acfBestPeriod)
                except:
                    break

                # Additional pre-classification
                if (i == 0 and BLSmaxPower < 100 and avgMaxSD < 7) or \
                        (acfMaxPower < 0.05 and avgMaxSD < 4) or BLSmaxPower < 60:
                    # No need to smooth attempt further, very unlikely to be obvious EBs.

                    if knownData == 1 and flag == 'EB':
                        title = 'Misclassified: ' + objName + "\n" + str(i) + "x smoothing" + "\n" + file

                        # Add to table
                        print("Adding to table.")
                        lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag,
                                             classif, rangeSD, file)

                        figName = 'missedEB_' + objName + '.png'

                        if i == 0:
                            plt.figure(figsize=(9, 9))
                            # Graph the light curve vs. smoothed light curve
                            print("Generating misclassified chart.")
                            makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux',
                                      'Light Curve for ' + objName, 'tab:purple', '.', .2)
                            plt.suptitle(title)
                            plt.savefig(os.path.join('misclassified', figName), orientation='landscape')
                            plt.close()
                        else:
                            plt.figure(figsize=(9, 9))
                            # Graph the light curve vs. smoothed light curve
                            print("Generating misclassified chart.")
                            plt.subplot(211)
                            makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux',
                                      'Light Curve for ' + objName, 'tab:purple', '.', .2)

                            plt.subplot(212)
                            makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                                      'Smoothed ' + str(i) + 'x light curve for ' + objName, 'tab:purple', '.', .2)

                            plt.suptitle(title)
                            plt.savefig(os.path.join('misclassified', figName), orientation='landscape')
                            plt.close()
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

                if knownData == 1 and ((flag == 'notEB' and classif == 'EB') or (i == 2 and flag != classif)):
                    plt.figure(figsize=(16, 16))
                    title = 'Misclassified: ' + objName + " Avg Max SD: " + str(avgMaxSD) + "\n" + str(i) + \
                            "x smoothing" + "\n" + file

                    if flag == 'EB':
                        figName = 'missedEB_' + objName + '.png'
                    else:
                        figName = 'misclassified_' + objName + '.png'

                    # Graph the light curve vs. smoothed light curve
                    print("Generating misclassified chart.")
                    plt.subplot(221)
                    makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux',
                              'Light Curve for ' + objName, 'tab:purple', '.', .2)

                    plt.subplot(222)
                    makegraph(curveData['TIME'], curveData['REL_FLUX'], 'BJD - 2457000 (days)', 'Relative Flux',
                              'Smoothed ' + str(i) + 'x light curve for ' + objName, 'tab:purple', '.', .2)

                    plt.subplot(223)
                    makegraph(acfPeriod, acfPower, 'Period', 'AutoCorr Power', 'ACF for ' + objName, 'tab:purple',
                              style='sm')
                    plt.scatter(acfBestPeriod, acfMaxPower, c='C1')
                    plt.text(acfBestPeriod, acfMaxPower, 'Per: ' + str(acfBestPeriod))

                    plt.subplot(224)
                    makegraph(BLSperiod, BLSpower, 'Period', 'BLS Power', 'BLS for ' + objName, 'tab:purple', style='sm')
                    plt.scatter(BLSbestPeriod, BLSmaxPower, c='C1')
                    plt.text(BLSbestPeriod, BLSmaxPower, 'Per: ' + str(BLSbestPeriod))

                    plt.suptitle(title)
                    plt.savefig(os.path.join('misclassified', figName), orientation='landscape')
                    plt.close()

                # Add to table
                print("Adding to table.")
                lcTable = addtotable(lcTable, objName, BLSmaxPower, acfMaxPower[0], maxSD, avgMaxSD, i, flag, classif,
                                     rangeSD, file)

                i += 1

        #if knownData == 0 and classif == 'EB':
        #    figName = objName + '.png'
        #    plt.figure(figsize=(9, 6))
        #    # Graph the light curve vs. smoothed light curve
        #    print("Generating misclassified chart.")
        #    makegraph(originalTime, originalFlux, 'BJD - 2457000 (days)', 'Relative Flux', 'Light Curve for ' + objName + '\n' + file,
        #              'tab:purple', '.', .2)
        #    plt.savefig(os.path.join('EB', figName), orientation='landscape')
        #    plt.close()
    print(objName + " complete.")

print('\nPlotting and classification complete.\n')
print('EBs found: ' + str(len(EBs)))
for EB in EBs:
    print(EB)

# Print table to file
print("\nPrint curve table to file.\n")
lcTable.to_csv('curvesTable.csv')

# Print results table to file
#print("Print results table to file.\n")
#grouped = lcTable.groupby(['Filename']).last()
#grouped.to_csv('resultsTable.csv')

if knownData == 1:
    # Extract table data necessary for graphs
    focusTable = grouped.filter(['Obj ID', 'BLS Max Pow', 'ACF Max Pow', 'Max SD', 'AvgMax SD', 'Times Smoothed', 'Flag', 'Filename'])

    ebFocus = grouped.loc[focusTable['Flag'] == 'EB'].copy()
    nonEBFocus = grouped.loc[focusTable['Flag'] != 'EB'].copy()

    # Generate graphs based on smoothing
    i = 0
    while i < 3:
        print("Generating charts for " + str(i) + "x smoothing.")
        filename = 'comps_' + str(i) + '.png'
        ebTable = ebFocus.loc[ebFocus['Times Smoothed'] == i].copy()
        otherTable = nonEBFocus.loc[nonEBFocus['Times Smoothed'] == i].copy()

        plt.figure(figsize=(9, 9))

        # ACF MP vs Max SD
        plt.subplot(211)
        makegraph(otherTable['AvgMax SD'], otherTable['ACF Max Pow'], 'AvgMax SD', 'ACF Max Power',
                  'ACF MP vs Max SD for ' + str(i) + 'x Smoothing', 'tab:gray', '^', 25)
        makegraph(ebTable['AvgMax SD'], ebTable['ACF Max Pow'], 'AvgMax SD', 'ACF Max Power',
                  'ACF MP vs Max SD for ' + str(i) + 'x Smoothing', 'tab:cyan', 'o', 20)

        # BLS MP vs Max SD
        plt.subplot(212)
        makegraph(otherTable['AvgMax SD'], otherTable['BLS Max Pow'], 'AvgMax SD', 'BLS Max Power',
                  'BLS MP vs Max SD for ' + str(i) + 'x Smoothing', 'tab:gray', '^', 25)
        makegraph(ebTable['AvgMax SD'], ebTable['BLS Max Pow'], 'AvgMax SD', 'BLS Max Power',
                  'BLS MP vs Max SD for ' + str(i) + 'x Smoothing', 'tab:cyan', 'o', 20)
        plt.axvline(10 - 2 * i, c='C9')
        y = startMP - dropMP * i
        plt.axhline(y, c='C9')

        plt.savefig(os.path.join('charts', filename), orientation='landscape')
        plt.close()

        i += 1

    # Confusion matrix
    print("Generating Confusion matrix.")
    confData = grouped.copy
    y_true = confData['Flag']
    y_pred = confData['Classification']
    labels = ['EB', 'Not EB']
    cf_matrix = confusion_matrix(y_true, y_pred)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    heatmap = sns.heatmap(cf_matrix, annot=True, xticklabels=labels, yticklabels=labels,
                          cmap='Blues')
    heatmap.figure.savefig(os.path.join('charts', 'confusionMatrix.png'))
    print("Confusion matrix complete.\n")

print("\nProcess complete.\n")

end = timer.time()
hours, rem = divmod(end - start, 3600)
minutes, seconds = divmod(rem, 60)
print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
