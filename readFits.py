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
    # plt.savefig(os.path.join('curves', figName), orientation='landscape')
    # plt.clf()

    # Lomb-Scargle Periodograms
    print("Generating Lomb-Scargle periodogram.")
    LS = LombScargle(curveData['TIME'], curveData['REL_FLUX'])  # , curveData['REL_FLUX_ERR'])
    frequency, power = LS.autopower(minimum_frequency=1 / 27, maximum_frequency=1 / .1)
    best_frequency = frequency[np.argmax(power)]

    LSmodel = LS.model(curveData['TIME'], best_frequency)
    plt.subplot(222)
    plt.plot(1 / frequency, power)
    plt.scatter(1 / best_frequency, np.max(power), c='C1')
    plt.xlabel('Period')
    # plt.xscale('log')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle for ' + objName)
    # plt.savefig(os.path.join('lombScargle', figName), orientation='landscape')
    # plt.clf()

    # Graph best fit
    # plt.plot(curveData['TIME'], LSmodel,
    #          label='L-S P=' + format(1. / best_frequency, '6.3f') + 'd, pk=' + format(np.nanmax(power), '6.3f'))
    # plt.savefig(os.path.join('powerSpectrum', figName), orientation='landscape')
    # plt.clf()

    # Autocorrelation Function using exoplanet.
    print("Generating ACF periodogram.")
    acf = xo.autocorr_estimator(curveData['TIME'].values, curveData['REL_FLUX'].values,
                                yerr=curveData['REL_FLUX_ERR'].values,
                                min_period=0.1, max_period=27, max_peaks=10)

    # smo = curveData['PDCSAP_FLUX'].rolling(128, center=True).median()
    acfPeriod = acf['autocorr'][0]
    acfPower = acf['autocorr'][1]

    acfPowerPd = pd.DataFrame(acfPower)
    acfLocalMaxima = acfPowerPd[(acfPowerPd.shift(1) < acfPowerPd) & (acfPowerPd.shift(-1) < acfPowerPd)]
    acfBestPower = np.max(acfLocalMaxima).values
    acfBestPeriod = acfPeriod[np.where(acfPower == acfBestPower)[0]]
    acfMeanPeriod = acfLocalMaxima.diff().mean
    # print(np.where(acfPower == acfBestPower)[0])
    # #print(acfBestPeriod)
    # print(acfBestPower)

    # plt.figure(figsize=(12, 9))
    plt.subplot(223)
    plt.plot(acfPeriod, acfPower)
    plt.scatter(acfBestPeriod, acfBestPower, c='C1')
    # plt.errorbar(curveData['TIME'], curveData['REL_FLUX'], yerr=curveData['REL_FLUX_ERR'], linestyle=None, alpha=0.15, label='PDC_FLUX')
    plt.xlabel('Period')
    plt.ylabel('AutoCorr Power')
    plt.title('ACF for ' + objName)
    # plt.savefig(os.path.join('acf', figName), orientation='landscape')
    # plt.close()

    # Box Least Squares
    print("Generating BLS periodogram.")
    model = BoxLeastSquares(curveData['TIME'].values, curveData['REL_FLUX'].values,
                            dy=curveData['REL_FLUX_ERR'].values)
    periodogram = model.power(period=acfPeriod[100:], duration=[0.05], objective='snr')
    BLSmaxPower = np.max(periodogram.power)
    BLSbestPeriod = periodogram.period[np.argmax(periodogram.power)]
    plt.subplot(224)
    plt.plot(periodogram.period, periodogram.power)
    plt.scatter(BLSbestPeriod, BLSmaxPower, c='C1')
    plt.xlabel('Period')
    plt.ylabel('Power')
    plt.title('BLS for ' + objName)
    # plt.savefig(os.path.join('bls', figName), orientation='landscape')
    plt.savefig(os.path.join('plots', figName), orientation='landscape')

    # Adjust the subplot layout
    plt.subplots_adjust(top=0.92, bottom=0.10, left=0.10, right=0.95, hspace=1,
                        wspace=1)
    plt.close()

    # Downsampling to prep for Altair graphing
    # Convert time into the necessary time series format for resampling.
    curveData.index = pd.to_timedelta(curveData.index, unit='T')
    res = '6T'  # New resolution, T represents minutes
    downsampledCurveData = curveData.resample(res).median()

    # Altair interactive graphing
    print("Generating interactive light curve.")
    chart = alt.Chart(downsampledCurveData).mark_circle(size=5).encode(
        alt.X('TIME', axis=alt.Axis(title='BJD - 2457000 (days)'), scale=alt.Scale(zero=False)),
        alt.Y('REL_FLUX', axis=alt.Axis(title='Relative Flux'), scale=alt.Scale(zero=False)),
        tooltip=['TIME', 'REL_FLUX']
    ).properties(title='Light Curve', width=750, height=500).interactive()

    saveFile = os.path.join('interactiveCurves', objName + '.html')
    chart.save(saveFile)
    print(objName + " complete.")
