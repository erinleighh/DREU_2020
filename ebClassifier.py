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


def classification(acfMP, lsMP, blsMP, sd):
    result = ''

    # Identify significant eclipses
    if (sd >= 6 and blsMP > 50) or (sd >= 5 and blsMP > 500):
        result = 'Preliminary Classification: EB, SD: ' + str(sd)
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
        return result
    elif sd >= 4 and lsMP > .2 and acfMP > .2:  # Add constraints to less significant eclipses
        result = 'Preliminary Classification: EB, SD: ' + str(sd)
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
        return result
    elif blsMP > 1000:
        result = 'Preliminary Classification: EB, SD: ' + str(sd)
        print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
        return result

    return result


def smoothing(time, relFlux, period):
    amp = relFlux[relFlux.where(relFlux > 1)].mean()
    omega = 2 * math.pi / period
    antiWave = np.sin(time.multiply(omega)).multiply(amp)
    print(len(relFlux))
    print(len(time))
    return relFlux.subtract(antiWave)


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
    period = periodogram.period
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    return period, power, bestPeriod, maxPower


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

    # Calculate z-score of all points, find outliers below the flux midpoint.
    z = np.abs(stats.zscore(curveData['REL_FLUX']))
    potentialEclipses = z[np.where(curveData['REL_FLUX'] < 1)[0]]
    maxSD = math.floor(np.max(potentialEclipses))

    # Classify based on outliers
    title = ''

    if maxSD < 3:
        plt.figure()
        figName = objName + '.png'
        title = 'Preliminary Classification: Not EB, Max SD: ' + str(maxSD)

        # Graph the light curve
        print("Generating light curve.")
        plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
        plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
        plt.ylabel('Relative Flux')
        plt.title('Light Curve for ' + objName)

        # Save figure
        plt.suptitle(title)
        plt.savefig(os.path.join('notEB', figName), orientation='landscape')
    else:
        # Figure to contain light curve and LS, ACF, and BLS periodograms.
        plt.figure(figsize=(16, 12))
        figName = objName + '.png'

        # Graph the light curve
        plt.subplot(221)
        print("Generating light curve.")
        plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
        plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
        plt.ylabel('Relative Flux')
        plt.title('Light Curve for ' + objName)

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
        title = classification(acfMaxPower, LSmaxPower, BLSmaxPower, maxSD)

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

    if title == 'Preliminary Classification: EB, SD: ' + str(maxSD):
        EBs.append(objName)

        # Save figure
        plt.suptitle(title)
        plt.savefig(os.path.join('EB', figName), orientation='landscape')

    if title == '':
        title = 'Preliminary Classification: Not EB, Max SD: ' + str(maxSD)

        # Save figure
        plt.suptitle(title)
        plt.savefig(os.path.join('notEB', figName), orientation='landscape')

    # Save figure
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
