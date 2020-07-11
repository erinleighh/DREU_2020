import glob
import os
import altair as alt
import pandas as pd
import numpy as np
import exoplanet as xo
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.timeseries import LombScargle, BoxLeastSquares


def classification(acfBP, blsBP, acfMP, lsMP, blsMP):
    ratio = acfBP / blsBP
    result = ''

    if ((round(ratio, 2) == 1) or (
            round(ratio / 2, 2) == 1)) and lsMP < 0.25 and acfMP > 0.1:  # Ratios of 1 or 2 correlated to EBs.
        if blsMP > 1000:
            result = 'Preliminary Classification: EB'
            print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')

        # if blsMP > 1000:
        #     if blsMP > 200:
        #         result = 'Preliminary Classification: EB'
        #         print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
        # else:
        #         if blsMP > 50 or lsMP > 0.65:
        #             result = 'Preliminary Classification: Possible EB'
        #             print(objName + ' NEEDS INSPECTED*************************')
        #         else:
        #             result = 'Preliminary Classification: Not EB'

    return result


def lombscargle(name, index, time, relFlux):
    LS = LombScargle(time, relFlux)
    frequency, power = LS.autopower(minimum_frequency=1 / 27, maximum_frequency=1 / .1)
    bestPeriod = 1 / frequency[np.argmax(power)]
    maxPower = np.max(power)
    period = 1 / frequency

    # plt.plot(period, power)
    # plt.scatter(bestPeriod, maxPower, c='C1')
    # plt.text(bestPeriod, maxPower, 'Per: ' + str(bestPeriod))
    # plt.xlabel('Period')
    # plt.ylabel('AutoCorr Power')
    # plt.title('LS for ' + name)
    # figName = name + '_' + str(index) + '.png'
    # plt.savefig(os.path.join('ls', figName), orientation='landscape')
    # plt.close()

    return period, power, bestPeriod, maxPower


def autocorrelationfn(name, index, time, relFlux, relFluxErr):
    acf = xo.autocorr_estimator(time.values, relFlux.values, yerr=relFluxErr.values,
                                min_period=0.1, max_period=27, max_peaks=10)

    period = acf['autocorr'][0]
    power = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(power)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    maxPower = np.max(acfLocalMaxima).values

    # plt.plot(period, power)
    # #plt.scatter(bestPeriod, maxPower, c='C1')
    # #plt.text(bestPeriod, maxPower, 'Per: ' + str(bestPeriod))
    # plt.xlabel('Period')
    # plt.ylabel('AutoCorr Power')
    # plt.title('ACF for ' + name)
    # figName = name + '_' + str(index) + '.png'
    # plt.savefig(os.path.join('acf', figName), orientation='landscape')
    # plt.close()

    bestPeriod = period[np.where(power == maxPower)[0]][0]
    peaks = acf['peaks'][0]['period']

    if len(acf['peaks']) > 0:
        window = int(peaks / np.abs(np.nanmedian(np.diff(time))) / 6.)
    else:
        window = 128

    return period, power, bestPeriod, maxPower, window


def boxleastsquares(name, index, time, relFlux, relFluxErr):
    model = BoxLeastSquares(time.values, relFlux.values, dy=relFluxErr.values)
    duration = [40 / 1440, 80 / 1440]
    periodogram = model.power(period=acfPeriod[np.where(acfPeriod > np.max(duration))[0]], duration=duration,
                              objective='snr')
    period = periodogram.period
    power = periodogram.power
    maxPower = np.max(periodogram.power)
    bestPeriod = periodogram.period[np.argmax(periodogram.power)]

    # plt.plot(period, power)
    # plt.scatter(bestPeriod, maxPower, c='C1')
    # plt.text(bestPeriod, maxPower, 'Per: ' + str(bestPeriod))
    # plt.xlabel('Period')
    # plt.ylabel('AutoCorr Power')
    # plt.title('BLS for ' + name)
    # figName = name + '_' + str(index) + '.png'
    # plt.savefig(os.path.join('bls', figName), orientation='landscape')
    # plt.close()

    return period, power, bestPeriod, maxPower


numTab = 11  # We have 11 fits files to read from.
lightCurves = [0] * numTab  # Store the light curves for all the tables.
path = "data"  # Hack for dealing with OS forward/back slash conflicts.
i = 0
EBs = []  # Store the objects classified as eclipsing binaries

for file in glob.glob(os.path.join(path, "*.fits")):
    fitsTable = fits.open(file, memmap=True)
    objName = fitsTable[0].header['OBJECT']
    print("\nReading in " + objName)
    curveTable = Table(fitsTable[1].data).to_pandas()
    curveData = curveTable.loc[curveTable['QUALITY'] == 0].dropna(subset=['TIME']).dropna(subset=['PDCSAP_FLUX']).copy()
    fluxMed = np.nanmedian(curveData['PDCSAP_FLUX'])
    time = curveData['TIME']
    relFlux = curveData['PDCSAP_FLUX'].div(fluxMed)
    relFluxErr = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

    # Plot with various axes and scales.
    plt.figure(figsize=(16, 18))

    i += 1
    figName = objName + '.png'

    # Classify & Smooth
    title = ''
    j = 0
    while title == '':
        # Graph the light curve
        if j == 0:
            print("Generating light curve.")
            plt.subplot(324)
            plt.scatter(time, relFlux, color='tab:purple', s=.1)
            plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
            plt.ylabel('Relative Flux')
            plt.title('Light Curve for ' + objName)
        else:
            if j == 1:
                print("Generating smoothed 1x light curve.")
                plt.subplot(325)
                plt.scatter(time, relFlux, color='tab:purple', s=.1)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                plt.title('Light Curve for ' + objName)
            else:
                print("Generating smoothed 1x light curve.")
                plt.subplot(326)
                plt.scatter(time, relFlux, color='tab:purple', s=.1)
                plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
                plt.ylabel('Relative Flux')
                plt.title('Light Curve for ' + objName)

        # Lomb-Scargle Periodograms
        print("Generating Lomb-Scargle periodogram.")
        LSperiod, LSpower, LSbestPeriod, LSmaxPower = lombscargle(objName, j, time, relFlux)

        # Autocorrelation Function using exoplanet.
        print("Generating ACF periodogram.")
        acfPeriod, acfPower, acfBestPeriod, acfMaxPower, s_window = autocorrelationfn(objName, j, time, relFlux, relFluxErr)

        # Box Least Squares
        print("Generating BLS periodogram.")
        BLSperiod, BLSpower, BLSbestPeriod, BLSmaxPower = boxleastsquares(objName, j, time, relFlux, relFluxErr)

        # Run classification
        title = classification(acfBestPeriod, BLSbestPeriod, acfMaxPower, LSmaxPower, BLSmaxPower)

        if title == 'Preliminary Classification: EB':
            EBs.append(objName)
        else:
            print("Performing smoothing on " + objName)
            relFlux = relFlux.rolling(s_window, center=True).median() / fluxMed
            relFluxErr = relFluxErr.rolling(s_window, center=True).median() / fluxMed

        if j == 3:
            title = 'Preliminary Classification: Not EB'
        j += 1

    # Plot final LS/ACF/BLS Fns

    # LS
    plt.subplot(321)
    plt.plot(LSperiod, LSpower)
    plt.scatter(LSbestPeriod, LSmaxPower, c='C1')
    plt.text(LSbestPeriod, LSmaxPower, 'Per: ' + str(LSbestPeriod))
    plt.xlabel('Period')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle for ' + objName)

    # ACF
    plt.subplot(322)
    plt.plot(acfPeriod, acfPower)
    plt.scatter(acfBestPeriod, acfMaxPower, c='C1')
    plt.text(acfBestPeriod, acfMaxPower, 'Per: ' + str(acfBestPeriod))
    plt.xlabel('Period')
    plt.ylabel('AutoCorr Power')
    plt.title('ACF for ' + objName)

    # BLS
    plt.subplot(323)
    plt.plot(BLSperiod, BLSpower)
    plt.scatter(BLSbestPeriod, BLSmaxPower, c='C1')
    plt.text(BLSbestPeriod, BLSmaxPower, 'Per: ' + str(BLSbestPeriod))
    plt.xlabel('Period')
    plt.ylabel('Power')
    plt.title('BLS for ' + objName)

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
