import glob
import os
import altair as alt
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

    # Identify significant eclipses
    if (sd >= 7 and blsMP > 100) or (sd >= 5 and blsMP > 500):
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


def boxleastsquares(time, relFlux, relFluxErr):
    model = BoxLeastSquares(time.values, relFlux.values, dy=relFluxErr.values)
    duration = [40 / 1440, 80 / 1440]
    periodogram = model.power(period=acfPeriod[np.where(acfPeriod > np.max(duration))[0]], duration=duration,
                              objective='snr')
    period = periodogram.periods
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    return period, power, bestPeriod, maxPower


lightCurves = []  # Initialize the array holding light curves
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

    originalFlux = curveData['REL_FLUX'].copy()
    originalTime = curveData['TIME'].copy()

    title = 'Unclassified'

    # Classify based on outliers
    i = 0
    while 'Unclassified' in title and i < 3:  # Potential to be an EB.
        bottom, top = 0, 0
        maxSD = findsd(curveData['REL_FLUX'], fluxMed)

        if maxSD < 3:  # Not an obvious EB, unlikely to be an EB.
            plt.figure()
            if i == 0:
                # Graph the light curve
                figName = objName + '.png'
                title = 'Preliminary Classification: Not EB, Max SD: ' + str(maxSD)
                print("Generating light curve.")
                plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                plt.title('Light Curve for ' + objName)
            else:
                plt.figure(figsize=(16, 12))
                figName = objName + '_' + str(i) + '.png'
                title = 'Preliminary Classification: Not EB, Max SD after ' + str(i) + 'x smoothing: ' + str(maxSD)
                # Graph the light curve
                print("Generating original light curve.")
                plt.subplot(211)
                plt.scatter(originalTime, originalFlux, color='tab:purple', s=.1)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                bottom, top = plt.ylim()
                plt.title('Light Curve for ' + objName)

                print('Generating smoothed ' + str(i) + 'x light curve.')
                plt.subplot(212)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
                plt.title('Smoothed ' + str(i) + 'x light curve for ' + objName)

            # Save figure
            plt.suptitle(title)
            plt.savefig(os.path.join('notEB', figName), orientation='landscape')
        else:
            # Figure to contain light curve and LS, ACF, and BLS periodograms.
            plt.figure(figsize=(16, 12))
            figName = objName + '_' + str(i) + '.png'

            if i == 0:
                print("Generating light curve.")
                plt.subplot(221)
                plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                bottom, top = plt.ylim()
                plt.title('Light Curve for ' + objName)
            else:
                print('Generating smoothed ' + str(i) + 'x light curve.')
                plt.subplot(221)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                bottom, top = plt.ylim()
                plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
                plt.title('Smoothed ' + str(i) + 'x light curve for ' + objName)

            # Lomb-Scargle Periodograms
            print("Generating Lomb-Scargle periodogram.")
            LSperiod, LSpower, LSbestPeriod, LSmaxPower = lombscargle(curveData['TIME'], curveData['REL_FLUX'])

            # Autocorrelation Function using exoplanet.
            print("Generating ACF periodogram.")
            acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(curveData['TIME'],
                                                                                          curveData['REL_FLUX'],
                                                                                          curveData['REL_FLUX_ERR'])

            # Box Least Squares
            print("Generating BLS periodogram.")
            BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(curveData['TIME'], curveData['REL_FLUX'],
                                                                              curveData['REL_FLUX_ERR'])

            # Run classification
            title = classification(BLSmaxPower, maxSD)

            # Plot LS/ACF/BLS Fns

            # LS
            plt.subplot(222)
            plt.plot(LSperiod, LSpower)
            plt.scatter(LSbestPeriod, LSmaxPower, c='C1')
            plt.text(LSbestPeriod, LSmaxPower, 'Per: ' + str(LSbestPeriod))
            plt.xlabel('Period')
            plt.ylabel('Power')
            plt.title('Lomb-Scargle for ' + objName)

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

            if 'Unclassified' in title:
                title = 'Unclassified, Max SD: ' + str(maxSD)
                # Perform Smoothing
                print("Performing smoothing on " + objName)
                smoothedFlux = curveData['REL_FLUX'].rolling(s_window, center=True).median()

                SOK = np.isfinite(smoothedFlux)

                newFlux = curveData['REL_FLUX'][SOK]-smoothedFlux[SOK]

                curveData['REL_FLUX'] = newFlux.copy()

                curveData = curveData.dropna(subset=['TIME']).dropna(subset=['REL_FLUX']).dropna(subset=['REL_FLUX_ERR']).copy()

                fluxMed = np.nanmedian(curveData['REL_FLUX'])

            if title == 'Preliminary Classification: EB, SD: ' + str(maxSD):
                title = 'Preliminary Classification: EB, SD: ' + str(maxSD) + "\n" + file
                EBs.append(objName)

                # Save figure
                plt.suptitle(title)
                plt.savefig(os.path.join('EB', figName), orientation='landscape')

                if i > 0:
                    plt.figure(figsize=(16, 12))
                    figName = objName + '.png'
                    title = 'Preliminary Classification: EB, SD: ' + str(maxSD) + "\n" + file
                    # Graph the light curve
                    print("Generating original light curve.")
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
                    plt.savefig(os.path.join('EB', figName), orientation='landscape')

            i += 1

            if 'Unclassified' in title and i == 3:
                if maxSD < 4:
                    title = 'Preliminary Classification: Not EB, Max SD after ' + str(i) + 'x smoothing: ' + str(
                        maxSD)

                    # Save figure
                    plt.suptitle(title)
                    plt.savefig(os.path.join('notEB', figName), orientation='landscape')
                else:
                    title = 'Preliminary Classification: Undetermined, Max SD: ' + str(maxSD)

                    # Save figure
                    plt.suptitle(title)
                    plt.savefig(os.path.join('und', figName), orientation='landscape')

        # Save figure
        plt.suptitle(title)
        plt.savefig(os.path.join('plots', figName), orientation='landscape')
        plt.close()

    # Downsampling to prep for Altair graphing
    # Convert time into the necessary time series format for resampling.
    # curveData.index = pd.to_timedelta(curveData.index, unit='T')
    # res = '6T'  # New resolution, T represents minutes
    # downsampledCurveData = curveData.resample(res).median()

    # Altair interactive graphing
    # print("Generating interactive light curve.")
    # chart = alt.Chart(downsampledCurveData).mark_circle(size=5).encode(
    #     alt.X('TIME', axis=alt.Axis(title='BJD - 2457000 (days)'), scale=alt.Scale(zero=False)),
    #     alt.Y('REL_FLUX', axis=alt.Axis(title='Relative Flux'), scale=alt.Scale(zero=False)),
    #     tooltip=['TIME', 'REL_FLUX']
    # ).properties(title='Light Curve', width=750, height=500).interactive()
    #
    # saveFile = os.path.join('interactiveCurves', objName + '.html')
    # chart.save(saveFile)
    print(objName + " complete.")

print('\nPlotting and classification complete.\n')
print('EBs found: ' + str(len(EBs)))
for EB in EBs:
    print(EB)
