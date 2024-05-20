import numpy as np
import librosa


def hz_to_audio(hz,sr=44100, hop_size=None):

    if hop_size:
        hz = np.repeat(hz, hop_size)

    phase = np.cumsum(2 * np.pi * hz / sr)
    return np.cos(phase)

def df_to_hz(df, new_sr=None):

    NATIVE_SR = 44100

    if new_sr:
        df['start_time'] = np.round(df['start_time'] / NATIVE_SR * new_sr).astype(int)
        df['end_time'] = np.round(df['end_time'] / NATIVE_SR * new_sr).astype(int)

    max_time = df['end_time'].max()


    f0_x = np.full(max_time + 1, np.nan)

    for _, row in df.iterrows():
        f0_x[row['start_time']:row['end_time']] = row['note']

    return librosa.midi_to_hz(f0_x)

def pow_to_db(x):
    return 10.0 * np.log10(x)