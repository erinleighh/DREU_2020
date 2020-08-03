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
from astropy.timeseries import LombScargle, BoxLeastSquares


def classification(blsMP, sd):
    result = 'Unclassified'

    # Line from SD 4, BLS 1000 to SD 15, BLS 100
    minblsMP = -81.818*sd + 1327.3

    # Identify significant eclipses
    # if (sd >= 7 and blsMP > 200) or (sd >= 5 and blsMP > 1000):
    if blsMP >= minblsMP:
        result = 'Preliminary Classification: EB, SD: ' + str(sd)
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
        return result

    return result


def findsd(relFlux, medianFlux):
    # Calculate z-score of all points, find outliers below the flux midpoint.
    z = np.abs(stats.zscore(relFlux))
    potentialEclipses = z[np.where(relFlux < min(1, medianFlux))[0]]
    maximumSD = math.floor(np.max(potentialEclipses))

    return maximumSD


def lombscargle(time, relFlux):
    LS = LombScargle(time, relFlux)
    frequency, power = LS.autopower(minimum_frequency=1 / 27, maximum_frequency=1 / .1)
    bestPeriod = 1 / frequency[np.argmax(power)]
    maxPower = np.max(power)
    period = 1 / frequency

    return period, power, bestPeriod, maxPower


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

    if len(acf['peaks']) > 0:
        window = int(peaks / np.abs(np.nanmedian(np.diff(time))) / 6.)
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


lightCurves = []  # Initialize the array holding light curves
path = "data"  # Folder containing fits files
EBs = []  # Store the objects classified as eclipsing binaries
ebFilenames = open("ebFilenames.txt", "a")
ebObjNames = open("ebObjNames.txt", "a")
plotting = 0

for file in glob.glob(os.path.join(path, "*.fits")):
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    print("\nReading in " + objName)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
    fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
    curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
    curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

    originalFlux = curveData['REL_FLUX'].copy()
    originalTime = curveData['TIME'].copy()

    title = 'Unclassified'

    # Classify based on outliers
    i = 0
    while 'Unclassified' in title and i < 10:  # Potential to be an EB.
        bottom, top = 0, 0
        maxSD = findsd(curveData['REL_FLUX'], fluxMed)

        if maxSD < 3:  # SD of 0 or 1 unlikely to be eclipse.
            title = 'Not EB'
        else:
            # Run ACF and BLS functions for classification

            try:
            # Autocorrelation Function using exoplanet.
                print("Generating ACF periodogram.")
                acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'],
                                                                                              curveData['REL_FLUX'],
                                                                                              curveData['REL_FLUX_ERR'])

                # Box Least Squares
                print("Generating BLS periodogram.")
                BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'], curveData['REL_FLUX'],
                                                                                  curveData['REL_FLUX_ERR'], acfBestPeriod)
            except:
                break
            # Run classification
            title = classification(BLSmaxPower, maxSD)

            if 'Unclassified' in title:
                # Perform Smoothing
                print("Performing smoothing on " + objName)
                smoothedFlux = curveData['REL_FLUX'].rolling(s_window, center=True).median()

                SOK = np.isfinite(smoothedFlux)

                newFlux = curveData['REL_FLUX'][SOK] - smoothedFlux[SOK]

                curveData['REL_FLUX'] = newFlux.copy()

                curveData = curveData.dropna(subset=['TIME']).dropna(subset=['REL_FLUX']).dropna(
                    subset=['REL_FLUX_ERR']).copy()

                fluxMed = np.nanmedian(curveData['REL_FLUX'])

            if title == 'Preliminary Classification: EB, SD: ' + str(maxSD):
                EBs.append(objName)  # Add to printout of EBs
                ebFilenames.write(file + "\n")
                ebObjNames.write(objName + "\n")
                if plotting:
                    # Make plot to hold original light curve, ACF, and BLS
                    plt.figure(figsize=(16, 12))
                    title = 'Preliminary Classification: EB, SD: ' + str(maxSD) + "\n" + file
                    figName = objName + '.png'

                    # Light Curve
                    print("Generating multi-plot figure.")
                    plt.subplot(211)
                    plt.scatter(originalTime, originalFlux, color='tab:purple', s=.1)
                    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                    plt.ylabel('Relative Flux')
                    bottom, top = plt.ylim()
                    plt.title('Light Curve for ' + objName)

                    # ACF
                    plt.subplot(223)
                    plt.plot(acfPeriod, acfPower)
                    plt.scatter(acfBestPeriod, acfMaxPower, c='C1')
                    plt.text(acfBestPeriod, acfMaxPower, 'Per: ' + str(acfBestPeriod))
                    plt.xlabel('Period')
                    plt.ylabel('AutoCorr Power')
                    plt.title('ACF for ' + objName)

                    # BLS
                    plt.subplot(224)
                    plt.plot(BLSperiod, BLSpower)
                    plt.scatter(BLSbestPeriod, BLSmaxPower, c='C1')
                    plt.text(BLSbestPeriod, BLSmaxPower, 'Per: ' + str(BLSbestPeriod))
                    plt.xlabel('Period')
                    plt.ylabel('Power')
                    plt.title('BLS for ' + objName)

                    plt.suptitle(title)
                    plt.savefig(os.path.join('EB', figName), orientation='landscape')
                    plt.close()

                # If smoothed, generate comparison between original light curve and smoothed curve
                if i > 0 and plotting:
                    plt.figure(figsize=(16, 12))
                    title = 'Preliminary Classification: EB, SD: ' + str(maxSD) + "\n" + file

                    # Graph the light curve vs. smoothed light curve
                    print("Generating original/smoothed light curve comparison.")
                    plt.subplot(211)
                    plt.scatter(originalTime, originalFlux, color='tab:purple', s=.1)
                    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                    plt.ylabel('Relative Flux')
                    bottom, top = plt.ylim()
                    plt.title('Light Curve for ' + objName)

                    plt.subplot(212)
                    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                    plt.ylabel('Relative Flux')
                    plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
                    plt.title('Smoothed ' + str(i) + 'x light curve for ' + objName)
                    plt.suptitle(title)
                    plt.savefig(os.path.join('smoothedEB', figName), orientation='landscape')
                    plt.close()
        i += 1

    print(objName + " complete.")
ebFilenames.close()
ebObjNames.close()

print('\nPlotting and classification complete.\n')
print('EBs found: ' + str(len(EBs)))
for EB in EBs:
    print(EB)
