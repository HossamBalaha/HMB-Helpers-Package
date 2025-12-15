import librosa
import numpy as np
import librosa.feature


class AudiosHelper(object):
  r'''
  AudiosHelper: Convenience methods for audio I/O and feature extraction.

  This helper provides wrappers around librosa and spafe feature extractors
  as well as parselmouth-based voice feature extraction. Methods try to be
  thin wrappers and keep the original behavior; spafe-based methods accept
  a sample-rate (sr / fs) and forward it to the underlying library.

  Notes:
    - Librosa is used for audio I/O and many feature extractions.
    - Spafe is used for additional feature extractions not covered by librosa.
    - You can install librosa and spafe via pip if not already installed: `pip install librosa spafe praat-parselmouth`
  '''

  def GetDuration(self, filePath):
    r'''
    Get the duration (in seconds) of an audio file.

    Parameters:
      filePath (str): Path to the audio file.

    Returns:
      float: Duration in seconds as reported by librosa.

    References:
      - Librosa get_duration documentation: https://librosa.org/doc/latest/generated/librosa.get_duration.html
    '''

    # Query librosa for the duration of the file.
    # Return result.
    return librosa.get_duration(filename=filePath)

  def GetSegmentDuration(self, y, sr, roundTo=3):
    r'''
    Compute the duration of an audio signal array.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz).
      roundTo (int): Number of decimals to round the result to (default 3).

    Returns:
      float: Rounded duration in seconds.

    References:
      - Librosa get_duration documentation: https://librosa.org/doc/latest/generated/librosa.get_duration.html
    '''

    # Compute and return the rounded duration using librosa.
    # Return result.
    return round(librosa.get_duration(y=y, sr=sr), roundTo)

  def Load(self, filePath, conversionType=True, offset=0, segmentDuration=1, isReversed=False):
    r'''
    Load an audio file segment using librosa.

    Parameters:
      filePath (str): Path to the audio file.
      conversionType (bool): If True, force mono. Matches librosa's `mono` parameter.
      offset (float): Start reading after this time (in seconds).
      segmentDuration (float): Duration to load (in seconds). If 0 or None, librosa loads full file.
      isReversed (bool): If True, reverse the returned signal (y[::-1]).

    Returns:
      tuple: (y, sr) where y is a 1-D numpy array and sr is the sampling rate (int).
        - numpy.ndarray: Audio time series.
        - int: Sampling rate of ``y``.

    References:
      - Librosa load documentation: https://librosa.org/doc/latest/generated/librosa.load.html
    '''

    if (segmentDuration is None or segmentDuration <= 0):
      # Load full file if duration is None or non-positive.
      segmentDuration = None

    # Load the requested portion of the file using librosa.
    y, sr = librosa.load(filePath, mono=conversionType, offset=offset, duration=segmentDuration)
    # Reverse the audio if requested.
    if (isReversed):
      # Reverse audio array.
      y = y[::-1]
    # Return the audio and sampling rate.
    # Return result.
    return y, sr

  def GetSTFT(self, y):
    r'''
    Compute the short-time Fourier transform (STFT) of a signal.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.

    Returns:
      numpy.ndarray: Complex-valued STFT matrix as returned by librosa.stft.

    References:
      - Librosa STFT: https://librosa.org/doc/latest/generated/librosa.stft.html
      - General STFT description: short-time Fourier transform literature.
    '''

    # Compute and return the STFT using librosa.
    # Return result.
    return librosa.stft(y)

  def GetAbsoluteSTFT(self, y):
    r'''
    Compute the magnitude (absolute value) spectrogram from the STFT.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.

    Returns:
      numpy.ndarray: Magnitude spectrogram (non-negative floats).

    References:
      - STFT magnitude and spectrogram conversions (librosa).
    '''

    # Compute and return the magnitude spectrogram.
    # Return result.
    return np.abs(self.GetSTFT(y))

  def GetHarmonicEffect(self, y):
    r'''
    Extract the harmonic component of a signal using librosa.effects.harmonic.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.

    Returns:
      numpy.ndarray: Harmonic component of the signal.

    References:
      - Librosa harmonic/percussive separation: https://librosa.org/doc/latest/generated/librosa.effects.harmonic.html
    '''

    # Extract and return the harmonic component.
    # Return result.
    return librosa.effects.harmonic(y)

  def GetPercussiveEffect(self, y):
    r'''
    Extract the percussive component of a signal using librosa.effects.percussive.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.

    Returns:
      numpy.ndarray: Percussive component of the signal.

    References:
      - Librosa harmonic/percussive separation: https://librosa.org/doc/latest/generated/librosa.effects.percussive.html
    '''

    # Extract and return the percussive component.
    # Return result.
    return librosa.effects.percussive(y)

  def GetSlaneyMFCC(self, y, sr=22050, nMFCC=None):
    r'''
    Compute Slaney-style MFCCs using librosa.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz). Default is 22050.
      nMFCC (int, optional): Number of MFCC coefficients to return. If None uses librosa default.

    Returns:
      numpy.ndarray: MFCC matrix (n_mfcc x frames).

    References:
      - MFCC background and Slaney implementation: Milner & Shao (2006) and Librosa MFCC docs.
      - Librosa mfcc: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
      - Milner, B., & Shao, X. (2006). Clean speech reconstruction from MFCC vectors and fundamental frequency using an integrated front-end.
    '''

    # Compute MFCC using Slaney configuration and return the result.
    if (nMFCC):
      # Return MFCC with specified number of coefficients.
      return librosa.feature.mfcc(y=y, sr=sr, dct_type=2, n_mfcc=nMFCC)
    # Return MFCC with default number of coefficients.
    return librosa.feature.mfcc(y=y, sr=sr, dct_type=2, n_mfcc=20)

  def GetMeanSlaneyMFCC(self, y, sr, nMFCC=None):
    r'''
    Compute the mean (per-coefficient) Slaney MFCC over time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz).
      nMFCC (int, optional): Number of MFCC coefficients to average.

    Returns:
      numpy.ndarray: 1-D array of length ``n_mfcc`` containing the mean over frames.

    References:
      - Librosa MFCC and averaging practices: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
    '''

    # Compute MFCCs and return their per-coefficient mean over time.
    # Return result.
    return np.mean(self.GetSlaneyMFCC(y, sr=sr, nMFCC=nMFCC).T, axis=0)

  def GetHtkMFCC(self, y, sr, nMFCC=None):
    r'''
    Compute HTK-style MFCCs using librosa with dct_type=3.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz).
      nMFCC (int, optional): Number of MFCC coefficients to return.

    Returns:
      numpy.ndarray: MFCC matrix (n_mfcc x frames).

    References:
      - HTK MFCC conventions (DCT type 3) and Librosa MFCC settings.
      - Librosa mfcc: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
    '''

    # Compute MFCC using HTK configuration and return the result.
    if (nMFCC):
      # Return HTK MFCC with specified number of coefficients.
      return librosa.feature.mfcc(y=y, sr=sr, dct_type=3, n_mfcc=nMFCC)
    # Return HTK MFCC with default number of coefficients.
    return librosa.feature.mfcc(y=y, sr=sr, dct_type=3)

  def GetMeanHtkMFCC(self, y, sr, nMFCC=None):
    r'''
    Return mean HTK MFCC coefficients across time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz).
      nMFCC (int, optional): Number of MFCC coefficients to average.

    Returns:
      numpy.ndarray: 1-D array with mean HTK MFCC coefficients.

    References:
      - HTK MFCC conventions and Librosa documentation.
    '''

    # Compute HTK MFCCs and return per-coefficient mean over time.
    # Return result.
    return np.mean(self.GetHtkMFCC(y, sr=sr, nMFCC=nMFCC).T, axis=0)

  def GetMeanChroma(self, y, sr):
    r'''
    Compute mean chroma_stft features across time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate of ``y`` (Hz).

    Returns:
      numpy.ndarray: 1-D array with 12 chroma mean values.

    References:
      - Chroma feature analysis: Ellis (2007). See Librosa chroma_stft docs.
      - Librosa chroma_stft: https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
    '''

    # Compute chroma STFT and return mean across frames.
    # Return result.
    return np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

  def GetMeanChromaSTFT(self, y, sr):
    r'''
    Compute mean chroma using the magnitude STFT as input.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean chroma vector.

    References:
      - Chroma via STFT: Librosa documentation and chroma literature.
      - Librosa chroma_stft: https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html
    '''

    # Compute magnitude STFT for chroma computation.
    stft = self.GetAbsoluteSTFT(y)
    # Compute and return chroma from magnitude STFT.
    # Return result.
    return np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

  def GetMeanChromaCqt(self, y, sr):
    r'''
    Compute mean chroma from the constant-Q transform.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean chroma vector from CQT.

    References:
      - Chroma CQT implementation: Librosa chroma_cqt docs.
      - Muller, M. & Ewert, S. (2011) Chroma Toolbox reference.
    '''

    # Compute and return chroma CQT mean across frames.
    # Return result.
    return np.mean(librosa.feature.chroma_cqt(y=y, sr=sr).T, axis=0)

  def GetMeanChromaCens(self, y, sr):
    r'''
    Compute chroma CENS mean across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean chroma CENS vector.

    References:
      - Chroma CENS method: Chroma Toolbox and related publications.
    '''

    # Compute and return chroma CENS mean across frames.
    # Return result.
    return np.mean(librosa.feature.chroma_cens(y=y, sr=sr).T, axis=0)

  def GetMeanMelSpectrogram(self, y, sr):
    r'''
    Compute mean Mel spectrogram across time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean Mel spectrogram vector across frames.

    References:
      - Librosa melspectrogram: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
    '''

    # Compute and return mean mel spectrogram over time.
    # Return result.
    return np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

  def GetMeanSpectralContrast(self, y, sr):
    r'''
    Compute mean spectral contrast across time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean spectral contrast vector across frames.

    References:
      - Spectral contrast: Jiang et al. (2002). See Librosa spectral_contrast docs.
      - Librosa spectral_contrast: https://librosa.org/doc/latest/generated/librosa.feature.spectral_contrast.html
    '''

    # Compute and return mean spectral contrast across frames.
    # Return result.
    return np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

  def GetMeanHarmonicTonnetz(self, y, sr):
    r'''
    Compute mean Tonnetz on the harmonic component.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean Tonnetz vector computed on harmonic component.

    References:
      - Tonnetz features: Harte et al. (2006). See Librosa tonnetz docs.
      - Librosa tonnetz: https://librosa.org/doc/latest/generated/librosa.feature.tonnetz.html
    '''

    # Extract harmonic component.
    harmonic = self.GetHarmonicEffect(y)
    # Compute and return Tonnetz mean on harmonic component.
    # Return result.
    return self.GetMeanTonnetz(harmonic, sr=sr)

  def GetMeanTonnetz(self, y, sr):
    r'''
    Compute mean Tonnetz features across time.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean Tonnetz vector across frames.

    References:
      - Tonnetz: Harte et al. (2006) and Librosa documentation.
    '''

    # Compute and return mean tonnetz across frames.
    # Return result.
    return np.mean(librosa.feature.tonnetz(y=y, sr=sr).T, axis=0)

  def GetMeanRMS(self, y, sr):
    r'''
    Compute mean root-mean-square energy across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean RMS energy per frame.

    References:
      - RMS energy and librosa RMS: https://librosa.org/doc/latest/generated/librosa.feature.rms.html
    '''

    # Compute and return mean RMS across frames.
    # Return result.
    return np.mean(librosa.feature.rms(y=y).T, axis=0)

  def GetMeanSpectralCentroid(self, y, sr):
    r'''
    Compute mean spectral centroid across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean spectral centroid per frame.

    References:
      - Spectral centroid and bandwidth: Klapuri & Davy (2007). See Librosa docs.
      - Librosa spectral_centroid: https://librosa.org/doc/latest/generated/librosa.feature.spectral_centroid.html
    '''

    # Compute and return mean spectral centroid across frames.
    # Return result.
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)

  def GetMeanSpectralBandwidth(self, y, sr):
    r'''
    Compute mean spectral bandwidth across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean spectral bandwidth per frame.

    References:
      - Spectral bandwidth discussion and Librosa docs.
      - Librosa spectral_bandwidth: https://librosa.org/doc/latest/generated/librosa.feature.spectral_bandwidth.html
    '''

    # Compute and return mean spectral bandwidth across frames.
    # Return result.
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)

  def GetMeanSpectralRolloff(self, y, sr):
    r'''
    Compute mean spectral rolloff across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean spectral rolloff per frame.

    References:
      - Spectral rolloff: Librosa documentation.
      - Librosa spectral_rolloff: https://librosa.org/doc/latest/generated/librosa.feature.spectral_rolloff.html
    '''

    # Compute and return mean spectral rolloff across frames.
    # Return result.
    return np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

  def GetMeanSpectralFlatness(self, y, sr):
    r'''
    Compute mean spectral flatness across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean spectral flatness per frame.

    References:
      - Spectral flatness literature and Librosa docs.
      - Dubnov et al. (2004) on spectral flatness.
    '''

    # Compute and return mean spectral flatness across frames.
    # Return result.
    return np.mean(librosa.feature.spectral_flatness(y=y).T, axis=0)

  def GetMeanZCR(self, y, sr):
    r'''
    Compute mean zero-crossing rate across frames.

    Parameters:
      y (numpy.ndarray): 1-D audio time series.
      sr (int): Sampling rate.

    Returns:
      numpy.ndarray: Mean zero-crossing rate per frame.

    References:
      - Zero-crossing Rate features and Librosa docs.
      - Librosa zero_crossing_rate: https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html
    '''

    # Compute and return mean zero-crossing rate across frames.
    # Return result.
    return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

  def GenerateScaledMelSpectrogram(self, y, sr, hopLength=512, nFFT=2048, numMels=128):
    r'''
    Create a log-scaled Mel spectrogram (dB) suitable for visualization or model input.

    Parameters:
      y (numpy.ndarray): Audio time series.
      sr (int): Sampling rate.
      hopLength (int): Hop length for STFT.
      nFFT (int): FFT size.
      numMels (int): Number of Mel bands to generate.

    Returns:
      numpy.ndarray: Mel spectrogram in decibels (shape: n_mels x frames).

    References:
      - Mel spectrogram and power_to_db: Librosa documentation.
      - Librosa melspectrogram: https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
      - Librosa power_to_db: https://librosa.org/doc/latest/generated/librosa.power_to_db.html
    '''

    # Compute mel spectrogram using librosa.
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hopLength, n_fft=nFFT, n_mels=numMels)
    # Convert the mel spectrogram to magnitude.
    spectrogram = np.abs(mel)
    # Convert power spectrogram to decibel units.
    melDB = librosa.power_to_db(spectrogram, ref=np.max)
    # Return the decibel mel spectrogram.
    # Return result.
    return melDB

  def GenerateSTFT(self, y, sr, hopLength=512, nFFT=2048):
    r'''
    Compute a log-amplitude STFT spectrogram.

    Parameters:
      y (numpy.ndarray): Audio time series.
      sr (int): Sampling rate.
      hopLength (int): Hop length for STFT.
      nFFT (int): FFT size.

    Returns:
      numpy.ndarray: Log-amplitude STFT spectrogram.

    References:
      - STFT and amplitude-to-db: Librosa documentation.
      - Librosa amplitude_to_db: https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html
    '''

    # Compute STFT using librosa.
    stft = librosa.core.stft(y, hop_length=hopLength, n_fft=nFFT)
    # Compute magnitude spectrogram from complex STFT.
    spectrogram = np.abs(stft)
    # Convert amplitude spectrogram to decibels.
    logSpectro = librosa.amplitude_to_db(spectrogram)
    # Return the log-amplitude spectrogram.
    # Return result.
    return logSpectro

  def GetBFCC(self, y, sr=16000):
    r'''
    Compute BFCC (bark-frequency cepstral coefficients) using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate (fs) forwarded to spafe.bfcc.

    Returns:
      numpy.ndarray: BFCC feature matrix.

    References:
      - BFCC paper: https://asmp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13636-017-0100-x
    '''

    from spafe.features.bfcc import bfcc

    # Compute and return BFCC features via spafe.
    # Return result.
    return bfcc(y, fs=sr, normalize=0)

  def GetGFCC(self, y, sr=16000):
    r'''
    Compute GFCC (gammatone-frequency cepstral coefficients) using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: GFCC feature matrix.

    References:
      - GFCC reference: https://www.researchgate.net/publication/309149564_Robust_Speaker_Verification_Using_GFCC_Based_i-Vectors
    '''

    from spafe.features.gfcc import gfcc

    # Compute and return GFCC features via spafe.
    # Return result.
    return gfcc(y, fs=sr, normalize=0)

  def GetLFCC(self, y, sr=16000):
    r'''
    Compute LFCC (linear-frequency cepstral coefficients) using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: LFCC feature matrix.

    References:
      - LFCC reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf
    '''

    from spafe.features.lfcc import lfcc

    # Compute and return LFCC features via spafe.
    # Return result.
    return lfcc(y, fs=sr, normalize=0)

  def GetLPC(self, y, sr=16000):
    r'''
    Compute LPC coefficients using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: LPC coefficient matrix.

    References:
      - Linear predictive coding (LPC) literature and implementations.
    '''

    from spafe.features.lpc import lpc

    # Compute and return LPC coefficients via spafe.
    # Return result.
    return lpc(y, fs=sr)

  def GetLPCC(self, y, sr=16000):
    r'''
    Compute LPCC coefficients using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: LPCC coefficient matrix.

    References:
      - LPCC and LP-based cepstral literature.
    '''

    from spafe.features.lpc import lpcc

    # Compute and return LPCC coefficients via spafe.
    # Return result.
    return lpcc(y, fs=sr, normalize=0)

  def GetMFCC(self, y, sr=16000):
    r'''
    Compute MFCC using spafe (not librosa's MFCC).

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: MFCC feature matrix from spafe.

    References:
      - MFCC literature and implementations. See Milner & Shao (2006) and Librosa docs.
    '''

    from spafe.features.mfcc import mfcc

    # Compute and return MFCC via spafe.
    # Return result.
    return mfcc(y, fs=sr, normalize=0)

  def GetIMFCC(self, y, sr=16000):
    r'''
    Compute IMFCC (inverse MFCC) using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: IMFCC feature matrix.

    References:
      - IMFCC reference literature and spafe implementation.
    '''

    from spafe.features.mfcc import imfcc

    # Compute and return IMFCC features via spafe.
    # Return result.
    return imfcc(y, fs=sr, normalize=0)

  def GetMSRCC(self, y, sr=16000):
    r'''
    Compute MSRCC features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: MSRCC feature matrix.

    References:
      - MSRCC reference: http://www.apsipa.org/proceedings/2018/pdfs/0001945.pdf
    '''

    from spafe.features.msrcc import msrcc

    # Compute and return MSRCC features via spafe.
    # Return result.
    return msrcc(y, fs=sr, normalize=0)

  def GetNGCC(self, y, sr=16000):
    r'''
    Compute NGCC features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: NGCC feature matrix.

    References:
      - NGCC references and related publications.
    '''

    from spafe.features.ngcc import ngcc

    # Compute and return NGCC features via spafe.
    # Return result.
    return ngcc(y, fs=sr, normalize=0)

  def GetPNCC(self, y, sr=16000):
    r'''
    Compute PNCC features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: PNCC feature matrix.

    References:
      - PNCC implementation reference: https://github.com/supikiti/PNCC/blob/master/pncc.py
    '''

    from spafe.features.pncc import pncc

    # Compute and return PNCC features via spafe.
    # Return result.
    return pncc(y, fs=sr, normalize=0)

  def GetPSRCC(self, y, sr=16000):
    r'''
    Compute PSRCC features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: PSRCC feature matrix.

    References:
      - PSRCC reference: http://www.apsipa.org/proceedings/2018/pdfs/0001945.pdf
    '''

    from spafe.features.psrcc import psrcc

    # Compute and return PSRCC features via spafe.
    # Return result.
    return psrcc(y, fs=sr, normalize=0)

  def GetPLP(self, y, sr):
    r'''
    Compute PLP features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: PLP feature matrix.

    References:
      - PLP literature and spafe usage notes.
    '''

    from spafe.features.rplp import plp

    # Compute and return PLP features via spafe.
    # Return result.
    return plp(y, fs=sr, normalize=0)

  def GetRPLP(self, y, sr):
    r'''
    Compute RPLP features using spafe.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: RPLP feature matrix.

    References:
      - RPLP references and spafe implementation.
    '''

    from spafe.features.rplp import rplp

    # Compute and return RPLP features via spafe.
    # Return result.
    return rplp(y, fs=sr, normalize=0)

  def GetMeanBFCC(self, y, sr=16000):
    r'''
    Return mean BFCC coefficients across time (per-coefficient mean).

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean BFCC vector.

    References:
      - BFCC paper: https://asmp-eurasipjournals.springeropen.com/track/pdf/10.1186/s13636-017-0100-x
    '''

    from spafe.features.bfcc import bfcc

    # Compute BFCC and return per-coefficient mean.
    # Return result.
    return np.mean(bfcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanGFCC(self, y, sr=16000):
    r'''
    Return mean GFCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean GFCC vector.

    References:
      - GFCC reference: https://www.researchgate.net/publication/309149564_Robust_Speaker_Verification_Using_GFCC_Based_i-Vectors
    '''

    from spafe.features.gfcc import gfcc

    # Compute GFCC and return per-coefficient mean.
    # Return result.
    return np.mean(gfcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanLFCC(self, y, sr=16000):
    r'''
    Return mean LFCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean LFCC vector.

    References:
      - LFCC reference: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf
    '''

    from spafe.features.lfcc import lfcc

    # Compute LFCC and return per-coefficient mean.
    # Return result.
    return np.mean(lfcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanLPC(self, y, sr=16000):
    r'''
    Return mean LPC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean LPC vector.

    References:
      - LPC refernece: https://superkogito.github.io/spafe/features/lpc.html
    '''

    from spafe.features.lpc import lpc

    # Compute LPC and return per-coefficient mean.
    # Return result.
    return np.mean(lpc(y, fs=sr)[0], axis=0)

  def GetMeanLPCC(self, y, sr=16000):
    r'''
    Return mean LPCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean LPCC vector.

    References:
      - LPCC literature on cepstral analysis.
    '''

    from spafe.features.lpc import lpcc

    # Compute LPCC and return per-coefficient mean.
    # Return result.
    return np.mean(lpcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanMFCC(self, y, sr=16000):
    r'''
    Return mean MFCC coefficients (from spafe-mfcc) across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean MFCC vector.

    References:
      - MFCC reference: https://spafe.readthedocs.io/en/latest/features/mfcc.html
      - Milner & Shao (2006) reference for MFCC reconstruction.
    '''

    from spafe.features.mfcc import mfcc

    # Compute MFCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(mfcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanIMFCC(self, y, sr=16000):
    r'''
    Return mean IMFCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean IMFCC vector.

    References:
      - IMFCC literature and spafe docs.
    '''

    from spafe.features.mfcc import imfcc

    # Compute IMFCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(imfcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanMSRCC(self, y, sr=16000):
    r'''
    Return mean MSRCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean MSRCC vector.

    References:
      - MSRCC reference: http://www.apsipa.org/proceedings/2018/pdfs/0001945.pdf
    '''

    from spafe.features.msrcc import msrcc

    # Compute MSRCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(msrcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanNGCC(self, y, sr=16000):
    r'''
    Return mean NGCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean NGCC vector.

    References:
      - NGCC references in the literature.
    '''

    from spafe.features.ngcc import ngcc

    # Compute NGCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(ngcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanPNCC(self, y, sr=16000):
    r'''
    Return mean PNCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean PNCC vector.

    References:
      - PNCC implementation: https://github.com/supikiti/PNCC/blob/master/pncc.py
    '''

    from spafe.features.pncc import pncc

    # Compute PNCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(pncc(y, fs=sr, normalize=0), axis=0)

  def GetMeanPSRCC(self, y, sr=16000):
    r'''
    Return mean PSRCC coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean PSRCC vector.

    References:
      - PSRCC reference: http://www.apsipa.org/proceedings/2018/pdfs/0001945.pdf
    '''

    from spafe.features.psrcc import psrcc

    # Compute PSRCC via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(psrcc(y, fs=sr, normalize=0), axis=0)

  def GetMeanPLP(self, y, sr):
    r'''
    Return mean PLP coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean PLP vector.

    References:
      - PLP literature.
    '''

    from spafe.features.rplp import plp

    # Compute PLP via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(plp(y, fs=sr, normalize=0), axis=0)

  def GetMeanRPLP(self, y, sr):
    r'''
    Return mean RPLP coefficients across time.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      numpy.ndarray: Mean RPLP vector.

    References:
      - RPLP and rasta-PLP references.
    '''

    from spafe.features.rplp import rplp

    # Compute RPLP via spafe and return per-coefficient mean.
    # Return result.
    return np.mean(rplp(y, fs=sr, normalize=0), axis=0)

  def GetAllMeanAudioFeatures(self, y, sr=16000):
    r'''
    Compute all mean audio features available in this class.

    Parameters:
      y (numpy.ndarray): Signal.
      sr (int): Sample rate.

    Returns:
      dict: Dictionary with mean feature vectors for all available features.
    '''

    d = {
      "MeanHTKMFCC"          : self.GetMeanHtkMFCC(y, sr),
      "MeanChroma"           : self.GetMeanChroma(y, sr),
      "MeanChromaSTFT"       : self.GetMeanChromaSTFT(y, sr),
      "MeanChromaCqt"        : self.GetMeanChromaCqt(y, sr),
      "MeanChromaCens"       : self.GetMeanChromaCens(y, sr),
      "MeanMelSpectrogram"   : self.GetMeanMelSpectrogram(y, sr),
      "MeanSpectralContrast" : self.GetMeanSpectralContrast(y, sr),
      "MeanHarmonicTonnetz"  : self.GetMeanHarmonicTonnetz(y, sr),
      "MeanTonnetz"          : self.GetMeanTonnetz(y, sr),
      "MeanRMS"              : self.GetMeanRMS(y, sr),
      "MeanSpectralCentroid" : self.GetMeanSpectralCentroid(y, sr),
      "MeanSpectralBandwidth": self.GetMeanSpectralBandwidth(y, sr),
      "MeanSpectralRolloff"  : self.GetMeanSpectralRolloff(y, sr),
      "MeanSpectralFlatness" : self.GetMeanSpectralFlatness(y, sr),
      "MeanZCR"              : self.GetMeanZCR(y, sr),
      "MeanBFCC"             : self.GetMeanBFCC(y, sr),
      "MeanGFCC"             : self.GetMeanGFCC(y, sr),
      "MeanLFCC"             : self.GetMeanLFCC(y, sr),
      "MeanLPC"              : self.GetMeanLPC(y, sr),
      "MeanLPCC"             : self.GetMeanLPCC(y, sr),
      "MeanMFCC"             : self.GetMeanMFCC(y, sr),
      # "MeanIMFCC"            : self.GetMeanIMFCC(y, sr),
      "MeanMSRCC"            : self.GetMeanMSRCC(y, sr),
      "MeanNGCC"             : self.GetMeanNGCC(y, sr),
      "MeanPNCC"             : self.GetMeanPNCC(y, sr),
      "MeanPSRCC"            : self.GetMeanPSRCC(y, sr),
      "MeanPLP"              : self.GetMeanPLP(y, sr),
      "MeanRPLP"             : self.GetMeanRPLP(y, sr),
    }
    return d  # Return the dictionary of all mean features.

  def ExtractAudioFeaturesViaParselmouth(self, voiceSample, f0Min, f0Max, unit):
    r'''
    Extract voice features (pitch, jitter, shimmer, HNR, MFCC means) using parselmouth (Praat bindings).

    Parameters:
      voiceSample (str or numpy.ndarray): Path to audio file or a compatible array accepted by parselmouth.Sound.
      f0Min (float): Minimum pitch (Hz) for pitch extraction.
      f0Max (float): Maximum pitch (Hz) for pitch extraction.
      unit (str): Unit used by Praat for mean/std extraction (e.g., 'Hertz').

    Returns:
      dict: Dictionary with keys for f0 mean/std, hnr, jitter/shimmer measures, duration and mfccMean coefficients.

    References:
      - Parselmouth (Praat bindings) documentation: https://parselmouth.readthedocs.io/en/stable/
    '''

    from parselmouth.praat import call

    # Create a parselmouth Sound object from the provided input.
    sound = parselmouth.Sound(voiceSample)

    # Convert the sound to a Praat pitch object using given bounds.
    pitch = call(sound, "To Pitch", 0.0, f0Min, f0Max)
    # Extract mean pitch using Praat call.
    f0Mean = call(pitch, "Get mean", 0, 0, unit)
    # Extract pitch standard deviation using Praat call.
    f0StdDeviation = call(pitch, "Get standard deviation", 0, 0, unit)

    # Estimate harmonicity (HNR) from the sound.
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0Min, 0.1, 1.0)
    # Get mean harmonicity (HNR) value.
    hnr = call(harmonicity, "Get mean", 0, 0)

    # Convert sound to a point process for jitter/shimmer measures.
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0Min, f0Max)
    # Compute various jitter measures using Praat point process APIs.
    jitterRelative = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitterAbsolute = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitterRap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitterPpq5 = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitterDDP = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    # Compute various shimmer measures using Praat APIs.
    shimmerRelative = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmerLocalDb = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmerApq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmerApq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    # Get number of voiced points in the point process.
    numPoints = call(pointProcess, "Get number of points")
    # Get total duration of the sound.
    duration = call(sound, "Get total duration")

    # Extract MFCC coefficients using parselmouth Sound API.
    mfccObject = sound.to_mfcc(number_of_coefficients=12)
    # Convert the MFCC object to a numpy array.
    mfcc = mfccObject.to_array()
    # Compute mean MFCC coefficients across time.
    mfccMean = list(np.mean(mfcc.T, axis=0))

    # Build the result dictionary with extracted features.
    R = {
      "f0Mean"         : f0Mean,
      "f0StdDeviation" : f0StdDeviation,
      "hnr"            : hnr,
      "jitterRelative" : jitterRelative,
      "jitterAbsolute" : jitterAbsolute,
      "jitterRap"      : jitterRap,
      "jitterPpq5"     : jitterPpq5,
      "jitterDDP"      : jitterDDP,
      "shimmerRelative": shimmerRelative,
      "shimmerLocalDb" : shimmerLocalDb,
      "shimmerApq3"    : shimmerApq3,
      "shimmerApq5"    : shimmerApq5,
      "numPoints"      : numPoints,
      "duration"       : duration,
    }
    # Append mean MFCC coefficients to the result dictionary with numbered keys.
    for i, coef in enumerate(mfccMean):
      R["mfccMean" + str(i)] = coef

    # Return the assembled feature dictionary.
    # Return result.
    return R

  def GetMelImage(self, y, sr=16000, numMels=128, hopLength=512, nFFT=2048, outFrames=128, axisLast=True):
    r'''
    Generate a Mel spectrogram image (2D numpy array) from audio time series.

    Parameters:
      y (numpy.ndarray): Audio time series.
      sr (int): Sampling rate.
      numMels (int): Number of Mel bands.
      hopLength (int): Hop length for STFT.
      nFFT (int): FFT size.
      outFrames (int): Desired number of output frames (time dimension).

    Returns:
      numpy.ndarray: Mel spectrogram image with shape (numMels, outFrames, 3) if axisLast is True, else (3, numMels, outFrames).
    '''

    # Generate Mel spectrogram in dB.
    melDB = self.GenerateScaledMelSpectrogram(y, sr, hopLength, nFFT, numMels)

    if (melDB.shape[1] < outFrames):
      # Pad with zeros if there are fewer frames than outFrames.
      padWidth = outFrames - melDB.shape[1]
      melDB = np.pad(melDB, ((0, 0), (0, padWidth)), mode="constant", constant_values=(melDB.min(),))
    elif (melDB.shape[1] > outFrames):
      # Truncate to outFrames if there are more frames.
      melDB = melDB[:, :outFrames]

    delta = librosa.feature.delta(melDB)
    delta2 = librosa.feature.delta(melDB, order=2)
    # Stack the Mel spectrogram and its deltas to create a 3-channel image.
    melImage = np.stack([melDB, delta, delta2], axis=-1)

    # Normalize the image to zero mean and unit variance.
    melImage = (melImage - np.mean(melImage)) / np.std(melImage)

    # Return the Mel spectrogram image.
    return melImage
