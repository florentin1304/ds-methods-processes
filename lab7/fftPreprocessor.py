import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft, rfftfreq
#import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
import scipy.fft as fft


class FourierPreprocessor(BaseEstimator, TransformerMixin):
  def __init__(self, length, sample_rate):
    self.length = length
    self.sample_rate = sample_rate

  def fit(self, x, y=None):
    return self

  def transform(self, X):
    X_ = X.copy()
    new_df = pd.DataFrame()
    for i in range(len(X_)):
      row = X_[i]
      #row = row.to_numpy()
      row = row[~(np.isnan(row))]

      wav_fft = rfft(row)

      PSD = wav_fft * np.conjugate(wav_fft) / len(row)
      data = PSD.real[:2000]
      data_series = pd.Series(PSD.real).to_frame().T
      new_df = pd.concat([new_df, data_series])
    new_df = new_df.fillna(0)
    #new_df.index = X_.index


    return new_df

  def fft_lowpass(self, s, threshold=0.1e5):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d= 1/self.sample_rate) #2e-2 / s.size)
    fourier[frequencies > threshold] = 0
    return fourier