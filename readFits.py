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
    fluxMed = curveData['PDCSAP_FLUX'].median()
    curveData['REL_FLUX'] = curveData['PDCSAP_FLUX'].div(fluxMed)
    curveData['REL_FLUX_ERR'] = curveData['PDCSAP_FLUX_ERR'].div(fluxMed)

    # Plot with various axes and scales.
    plt.figure(figsize=(16, 12))

    # matplotlib graphing
    print("Generating light curve.")
    plt.subplot(221)
    plt.scatter(curveData['TIME'], curveData['REL_FLUX'], color='tab:purple', s=.1)
    plt.xlabel('BJD - 2457000 (days)')  # BJD Julian corrected for elliptical orbit.
    plt.ylabel('Relative Flux')
    plt.title('Light Curve for ' + objName)

    i += 1
    figName = objName + '.png'

    # Lomb-Scargle Periodograms
    print("Generating Lomb-Scargle periodogram.")
    LS = LombScargle(curveData['TIME'], curveData['REL_FLUX'])  # , curveData['REL_FLUX_ERR'])
    LSfrequency, LSpower = LS.autopower(minimum_frequency=1 / 27, maximum_frequency=1 / .1)
    best_frequency = LSfrequency[np.argmax(LSpower)]
    LSmodel = LS.model(curveData['TIME'], best_frequency)

    LSbestPeriod = 1 / best_frequency
    LSmaxPower = np.max(LSpower)
    plt.subplot(222)
    plt.plot(1 / LSfrequency, LSpower)
    plt.scatter(LSbestPeriod, LSmaxPower, c='C1')
    plt.text(LSbestPeriod, LSmaxPower, 'Per: ' + str(LSbestPeriod))
    plt.xlabel('Period')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle for ' + objName)

    # Graph best fit
    # plt.plot(curveData['TIME'], LSmodel,
    #          label='L-S P=' + format(1. / best_frequency, '6.3f') + 'd, pk=' + format(np.nanmax(LSpower), '6.3f'))
    # plt.savefig(os.path.join('powerSpectrum', figName), orientation='landscape')
    # plt.clf()

    # Autocorrelation Function using exoplanet.
    print("Generating ACF periodogram.")
    acf = xo.autocorr_estimator(curveData['TIME'].values, curveData['REL_FLUX'].values,
                                yerr=curveData['REL_FLUX_ERR'].values,
                                min_period=0.1, max_period=27, max_peaks=10)

    acfPeriod = acf['autocorr'][0]
    acfPower = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(acfPower)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    acfBestPower = np.max(acfLocalMaxima).values
    acfBestPeriod = acfPeriod[np.where(acfPower == acfBestPower)[0]][0]
    acfMeanPeriod = acfLocalMaxima.diff().mean

    # plt.figure(figsize=(12, 9))
    plt.subplot(223)
    plt.plot(acfPeriod, acfPower)
    plt.scatter(acfBestPeriod, acfBestPower, c='C1')
    plt.text(acfBestPeriod, acfBestPower, 'Per: ' + str(acfBestPeriod))
    # plt.errorbar(curveData['TIME'], curveData['REL_FLUX'], yerr=curveData['REL_FLUX_ERR'], linestyle=None,
    # alpha=0.15, label='PDC_FLUX')
    plt.xlabel('Period')
    plt.ylabel('AutoCorr Power')
    plt.title('ACF for ' + objName)

    # Box Least Squares
    print("Generating BLS periodogram.")
    model = BoxLeastSquares(curveData['TIME'].values, curveData['REL_FLUX'].values,
                            dy=curveData['REL_FLUX_ERR'].values)
    duration = [40 / 1440, 80 / 1440, 0.10]
    periodogram = model.power(period=acfPeriod[np.where(acfPeriod > np.max(duration))[0]], duration=duration,
                              objective='snr')
    BLSmaxPower = np.max(periodogram.power)
    BLSbestPeriod = periodogram.period[np.argmax(periodogram.power)]
    plt.subplot(224)
    plt.plot(periodogram.period, periodogram.power)
    plt.scatter(BLSbestPeriod, BLSmaxPower, c='C1')
    plt.text(BLSbestPeriod, BLSmaxPower, 'Per: ' + str(BLSbestPeriod))
    plt.xlabel('Period')
    plt.ylabel('Power')
    plt.title('BLS for ' + objName)

    # Classify
    ratio = acfBestPeriod / BLSbestPeriod

    if ((round(ratio, 2) == 1) or (round(ratio / 2, 2) == 1)) and LSmaxPower < 0.75:  # Ratios of 1 or 2 correlated to EBs.

        if acfBestPower > 0.1 or BLSmaxPower > 1750:
            if BLSmaxPower > 200:
                title = 'Preliminary Classification: EB'
                EBs.append(objName)
                print(objName + ' IS CLASSIFIED AS AN ECLIPSING BINARY****')
            else:
                if BLSmaxPower > 50 or LSmaxPower > 0.65:
                    title = 'Preliminary Classification: Possible EB'
                    EBs.append(objName + '*')
                    print(objName + ' NEEDS INSPECTED*************************')
                else:
                    title = 'Preliminary Classification: Not EB'
        else:
            title = 'Preliminary Classification: Not EB'

    else:
        title = 'Preliminary Classification: Not EB'

    # Save 2x2 figure
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
