import numpy as np
import librosa

def hz_to_audio(hz,sr=44100):
    phase = np.cumsum(2 * np.pi * hz / sr)
    return np.cos(phase)

def df_to_hz(df):
    max_time = df['end_time'].max()
    f0_x = np.full(max_time + 1, np.nan)

    for _, row in df.iterrows():    
        f0_x[row['start_time']:row['end_time']] = row['note']

    return librosa.midi_to_hz(f0_x)