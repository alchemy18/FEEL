from sklearn.preprocessing import MinMaxScaler
from scipy.stats import linregress
from scipy import signal
from scipy.signal import butter
from itertools import chain
import pandas as pd
import numpy as np
import pickle
import cvxEDA.src.cvxEDA as cvxEDA
import pywt
import plotly.graph_objects as go
import plotly.express as px
import xgboost
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator

# class CustomXGBClassifier(xgboost.XGBClassifier, BaseEstimator):
#     def __sklearn_tags__(self):
#         return {"estimator_type": "classifier"}



'''Filter EDA data '''
def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    # Normalization of the cutoff signal
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

''' Apply the filter designed before '''
def butter_lowpass_filter_filtfilt(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

''' Decompose EDA data'''
def decomposition(eda, Fs=4):
    """
    Decomposes the EDA signal into phasic and tonic components using cvxEDA.
    
    Args:
        eda (array-like): The EDA signal to decompose.
        Fs (int): Sampling frequency (default: 4Hz).
    
    Returns:
        dict: A dictionary containing the phasic and tonic components.
    """
    y = np.array(eda)
    yn = (y - y.mean()) / y.std()
    # cvxEDA generates multiple outputs as a generator
    results = cvxEDA.cvxEDA(yn, 1.0 / Fs)
    
    # Convert generator to a list
    r, p, t, l, d, e, obj = list(results)
    
    return {
        "phasic": np.array(p).ravel(),
        "tonic": np.array(t).ravel()
    }

def preprocess_eda_signals(df):
    """
    Preprocesses the EDA data by filtering, decomposing, and calculating wavelets.
    
    Args:
        df (pd.DataFrame): DataFrame with a column named 'EDA'.
    
    Returns:
        pd.DataFrame: The same DataFrame with new columns for filtered EDA,
                      phasic/tonic components, and wavelet features.
    """
    # Apply low-pass Butterworth filter
    df['EDA_Filtered'] = df['Data']
    
    # Decompose EDA signal
    decomposition_results = decomposition(df['EDA_Filtered'], 4)
    df['EDA_Phasic'] = decomposition_results['phasic']
    df['EDA_Tonic'] = decomposition_results['tonic']
    
    # Calculate wavelets
    _, dtw3, dtw2, dtw1 = pywt.wavedec(df['EDA_Filtered'], 'Haar', level=3)
    
    # Repeat wavelet coefficients to match data length
    dtw3_duplicates = list(np.repeat(dtw3, 8))
    dtw2_duplicates = list(np.repeat(dtw2, 4))
    dtw1_duplicates = list(np.repeat(dtw1, 2))
    
    df['Wavelet3'] = dtw3_duplicates[:len(df)]
    df['Wavelet2'] = dtw2_duplicates[:len(df)]
    df['Wavelet1'] = dtw1_duplicates[:len(df)]
    
    # print("1. EDA data preprocessing completed")
    
    return df

def segment_eda_data(df, window_size = 60):
    """
    Segments the EDA signal into 5-second non-overlapping segments.
    The sampling frequency of our sensor was 4Hz, so the window size for segmenting is 20 samples.
    
    INPUT:
        df: DataFrame with columns named:
            'EDA_Filtered', 'EDA_Phasic', 'Wavelet1', 'Wavelet2', 'Wavelet3', 'Time'
    
    OUTPUT:
        segmented_df: New DataFrame with segmented data. Each cell contains an array of 20 values.
    """
    # Define window size (5 seconds * 4Hz = 20 samples)
    
    # Reshape each column into non-overlapping windows
    eda = df['EDA_Filtered'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    eda_phasic = df['EDA_Phasic'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    time = df['Time'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    wavelet3 = df['Wavelet3'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    wavelet2 = df['Wavelet2'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    wavelet1 = df['Wavelet1'].values[:len(df) - len(df) % window_size].reshape(-1, window_size)
    
    # Create a new DataFrame with segmented data
    segmented_df = pd.DataFrame({
        'EDA': list(eda),
        'EDA_Phasic': list(eda_phasic),
        'Time': list(time),
        'Wavelet3': list(wavelet3),
        'Wavelet2': list(wavelet2),
        'Wavelet1': list(wavelet1)
    })
    
    # print("2. EDA data segmentation completed")
    
    return segmented_df

def compute_statistical_wavelet(df):
    """
    Computes statistical and wavelet features for each preprocessed component of the EDA signal.
    Features are calculated over 5-second non-overlapping segments.
    
    INPUT:
        df: DataFrame with columns named:
            'EDA', 'EDA_Phasic', 'Time', 'Wavelet1', 'Wavelet2', 'Wavelet3'
    
    OUTPUT:
        df: Returns the same DataFrame with new statistical features for each column.
    """
    columns = ['EDA', 'EDA_Phasic', 'Wavelet1', 'Wavelet2', 'Wavelet3']

    for col in columns:
        data = df[col]
        name = col
        time = np.arange(1, len(data[0]) + 1)  # Ensure time matches the length of each segment

        # Initialize lists to store features
        medians, means, stds, variances, mins, maxs = [], [], [], [], [], []
        fdmeans, sdmeans, fdstds, sdstds, dranges, slopes = [], [], [], [], [], []

        # Compute features for each segment
        for i in range(len(data)):
            eda = data[i]
            fd = np.gradient(eda)  # First derivative
            sd = np.gradient(fd)  # Second derivative

            # Append features
            fdmeans.append(np.mean(fd))
            sdmeans.append(np.mean(sd))
            fdstds.append(np.std(fd))
            sdstds.append(np.std(sd))
            dranges.append(np.max(eda) - np.min(eda))
            medians.append(np.median(eda))
            means.append(np.mean(eda))
            stds.append(np.std(eda))
            variances.append(np.var(eda))
            mins.append(np.min(eda))
            maxs.append(np.max(eda))

            # Linear regression on time vs. EDA values
            slope, intercept, r_value, p_value, std_err = linregress(time, eda)
            slopes.append(slope)

        # Add features to DataFrame
        df[name + '_median'] = medians
        df[name + '_mean'] = means
        df[name + '_std'] = stds
        df[name + '_var'] = variances
        df[name + '_slope'] = slopes
        df[name + '_min'] = mins
        df[name + '_max'] = maxs
        df[name + '_fdmean'] = fdmeans
        df[name + '_fdstd'] = fdstds
        df[name + '_sdmean'] = sdmeans
        df[name + '_sdstd'] = sdstds
        df[name + '_drange'] = dranges

    # print("3. Statistical and wavelets feature extraction completed")
    return df

def findPeaks(data, offset, start_WT, end_WT, thres=0,SAMPLE_RATE = 4):
    '''
        This function finds the peaks of an EDA signal and returns basic properties.
        Also, peak_end is assumed to be no later than the start of the next peak.
        
        ********* INPUTS **********
        data:        DataFrame with EDA as one of the columns and indexed by a datetimeIndex
        offset:      the number of rising samples and falling samples after a peak needed to be counted as a peak
        start_WT:    maximum number of seconds before the apex of a peak that is the "start" of the peak
        end_WT:      maximum number of seconds after the apex of a peak that is the "rec.t/2" of the peak, 50% of amp
        thres:       the minimum uS change required to register as a peak, defaults as 0 (i.e. all peaks count)
        sampleRate:  number of samples per second, default=8
        
        ********* OUTPUTS **********
        peaks:               list of binary, 1 if apex of SCR
        peak_start:          list of binary, 1 if start of SCR
        peak_start_times:    list of strings, if this index is the apex of an SCR, it contains datetime of start of peak
        peak_end:            list of binary, 1 if rec.t/2 of SCR
        peak_end_times:      list of strings, if this index is the apex of an SCR, it contains datetime of rec.t/2
        amplitude:           list of floats,  value of EDA at apex - value of EDA at start
        max_deriv:           list of floats, max derivative within 1 second of apex of SCR
    '''
    
    sampleRate = SAMPLE_RATE
    
    EDA_deriv = data['EDA_Phasic'][1:].values - data['EDA_Phasic'][:-1].values
    peaks = np.zeros(len(EDA_deriv))
    peak_sign = np.sign(EDA_deriv)
    for i in range(int(offset), int(len(EDA_deriv) - offset)):
        if peak_sign[i] == 1 and peak_sign[i + 1] < 1:
            peaks[i] = 1
            for j in range(1, int(offset)):
                if peak_sign[i - j] < 1 or peak_sign[i + j] > -1:
                    peaks[i] = 0
                    break

    # Finding start of peaks
    peak_start = np.zeros(len(EDA_deriv))
    peak_start_times = [''] * len(data)
    max_deriv = np.zeros(len(data))
    rise_time = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            temp_start = max(0, i - sampleRate)
            max_deriv[i] = max(EDA_deriv[temp_start:i])
            start_deriv = .01 * max_deriv[i]

            found = False
            find_start = i
            # has to peak within start_WT seconds
            while found == False and find_start > (i - start_WT * sampleRate):
                if EDA_deriv[find_start] < start_deriv:
                    found = True
                    peak_start[find_start] = 1
                    peak_start_times[i] = data.index[find_start]
                    rise_time[i] = get_seconds_and_microseconds(data.index[i] - pd.to_datetime(peak_start_times[i]))

                find_start = find_start - 1

        # If we didn't find a start
            if found == False:
                peak_start[i - start_WT * sampleRate] = 1
                peak_start_times[i] = data.index[i - start_WT * sampleRate]
                rise_time[i] = start_WT

            # Check if amplitude is too small
            if thres > 0 and (data['EDA_Phasic'].iloc[i] - data['EDA_Phasic'][peak_start_times[i]]) < thres:
                peaks[i] = 0
                peak_start[i] = 0
                peak_start_times[i] = ''
                max_deriv[i] = 0
                rise_time[i] = 0

    # Finding the end of the peak, amplitude of peak
    peak_end = np.zeros(len(data))
    peak_end_times = [''] * len(data)
    amplitude = np.zeros(len(data))
    decay_time = np.zeros(len(data))
    half_rise = [''] * len(data)
    SCR_width = np.zeros(len(data))

    for i in range(0, len(peaks)):
        if peaks[i] == 1:
            peak_amp = data['EDA_Phasic'].iloc[i]
            start_amp = data['EDA_Phasic'][peak_start_times[i]]
            amplitude[i] = peak_amp - start_amp

            half_amp = amplitude[i] * .5 + start_amp

            found = False
            find_end = i
            # has to decay within end_WT seconds
            while found == False and find_end < (i + end_WT * sampleRate) and find_end < len(peaks):
                if data['EDA_Phasic'].iloc[find_end] < half_amp:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                    decay_time[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[i])

                    # Find width
                    find_rise = i
                    found_rise = False
                    while found_rise == False:
                        if data['EDA_Phasic'].iloc[find_rise] < half_amp:
                            found_rise = True
                            half_rise[i] = data.index[find_rise]
                            SCR_width[i] = get_seconds_and_microseconds(pd.to_datetime(peak_end_times[i]) - data.index[find_rise])
                        find_rise = find_rise - 1

                elif peak_start[find_end] == 1:
                    found = True
                    peak_end[find_end] = 1
                    peak_end_times[i] = data.index[find_end]
                find_end = find_end + 1

            # If we didn't find an end
            if found == False:
                min_index = np.argmin(data['EDA_Phasic'].iloc[i:(i + end_WT * sampleRate)].tolist())
                peak_end[i + min_index] = 1
                peak_end_times[i] = data.index[i + min_index]

    peaks = np.concatenate((peaks, np.array([0])))
    peak_start = np.concatenate((peak_start, np.array([0])))
    max_deriv = max_deriv * sampleRate  # now in change in amplitude over change in time form (uS/second)

    return peaks, peak_start, peak_start_times, peak_end, peak_end_times, amplitude, max_deriv, rise_time, decay_time, SCR_width, half_rise


def get_seconds_and_microseconds(pandas_time):
    return pandas_time.seconds + pandas_time.microseconds * 1e-6


def compute_peaks_features(df, SAMPLE_RATE):
    """
    This function computes peak features for each 5-second window in the EDA signal.
    
    INPUT:
        df: DataFrame with columns 'EDA_Phasic' and 'Time'
    
    OUTPUT:
        df: Updated DataFrame with new columns containing peak features.
    """
    thresh = 0.01
    offset = 1
    start_WT = 3
    end_WT = 10

    data = df.EDA
    times = df.Time

    peaks, rise_times, max_derivs, amps, decay_times, SCR_widths, aucs = [], [], [], [], [], [], []

    for i in range(len(data)):
        data_df = pd.DataFrame(columns=["EDA_Phasic", "Time"])
        data_df.EDA_Phasic = pd.Series(list(data[i]))

        # Ensure times[i] is a single timestamp
        start_time = pd.Timestamp(times[i][0]) if isinstance(times[i], (list, np.ndarray)) else pd.Timestamp(times[i])

        # Generate time range matching the length of data_df
        num_rows = len(data_df)
        data_df.Time = pd.date_range(start=start_time, periods=num_rows, freq='250ms')

        data_df.set_index(pd.DatetimeIndex(data_df['Time']), inplace=True)

        # Call findPeaks to compute peak features
        returnedPeakData = findPeaks(data_df, offset * SAMPLE_RATE, start_WT, end_WT, thresh, SAMPLE_RATE)
        result_df = pd.DataFrame(columns=["peaks", "amp", "max_deriv", "rise_time", "decay_time", "SCR_width"])
        result_df['peaks'] = returnedPeakData[0]
        result_df['amp'] = returnedPeakData[5]
        result_df['max_deriv'] = returnedPeakData[6]
        result_df['rise_time'] = returnedPeakData[7]
        result_df['decay_time'] = returnedPeakData[8]
        result_df['SCR_width'] = returnedPeakData[9]

        featureData = result_df[result_df.peaks == 1][['peaks', 'rise_time', 'max_deriv', 'amp', 'decay_time', 'SCR_width']]

        # Replace 0s with NaN for invalid values
        featureData[['SCR_width', 'decay_time']] = featureData[['SCR_width', 'decay_time']].replace(0, np.nan)
        featureData['AUC'] = featureData['amp'] * featureData['SCR_width']

        peaks.append(len(featureData))
        amps.append(result_df[result_df.peaks != 0.0].amp.mean())
        max_derivs.append(result_df[result_df.peaks != 0.0].max_deriv.mean())
        rise_times.append(result_df[result_df.peaks != 0.0].rise_time.mean())
        decay_times.append(featureData[featureData.peaks != 0.0].decay_time.mean())
        SCR_widths.append(featureData[featureData.peaks != 0.0].SCR_width.mean())
        aucs.append(featureData[featureData.peaks != 0.0].AUC.mean())

    df['peaks_p'] = peaks
    df['rise_time_p'] = rise_times
    df['max_deriv_p'] = max_derivs
    df['amp_p'] = amps
    df['decay_time_p'] = decay_times
    df['SCR_width_p'] = SCR_widths
    df['auc_p'] = aucs

    # print("4. Peaks feature extraction completed")

    return df

def remove_flat_responses(df):
    '''
    This function computes the peaks features for each 5 second window in the EDA signal.
    
    INPUT:
        df:        requires a dataframe with the calculated EDA slope feature
        
    OUTPUT:
        df:        returns a dataframe with only the 5-second windows that are not flat responses
    '''
    
    eda_flats = df.EDA_slope.between(-0.002, 0.002)
    df['Flat'] = eda_flats.values
    df['Flat'] = df.Flat.astype(int).values
    df_wo_flat = df[df.Flat == 0]
    
    # print("5. Flat responses removed")
        
    return df_wo_flat

def predict_shape_artifacts(features, df):
    '''
    This function computes whether a 5-second EDA segment is an artifact or not 
    
    INPUT:
        features:      list of features used as input in the model
        df:            requires a dataframe with columns 'EDA_Phasic' and 'Time'
        
    OUTPUT:
        df:            returns the same dataframe with new columnn "Artifact" that contains 
        a value of 1 when an artifact is predected and 0 otherwise.
    
    '''
    df = df.fillna(-1)

    # Normalize the features before providing as input to the model
#     scaler = MinMaxScaler()
#     for i in features:
#         df[i] = scaler.fit_transform(df[[i]])

    # Select only the features that have been used in the paper
    df_subselect = df[features]
    test_data = df_subselect.values

    # Load the trained model
    model = xgboost.XGBClassifier()
    model.load_model('SA_Detection.json')
    # model = pickle.load(open('SA_Detection.sav', 'rb'))

    # Use the loaded model to find artifacts 
    results = model.predict(test_data)

    # Create a new column that shows whether the window contains an artifact (1) or not (0)
    df['Artifact'] =list(results)
    
    # print("6. EDA artifacts detection completed")
    
    return df

def label_artifacts(database_wo_flats_artifacts, database):
    """
    This function adds the "Artifacts" column to the initial dataframe provided.
    
    INPUT:
        database_wo_flats_artifacts: DataFrame without flat responses
        database: Initial DataFrame with EDA and Time columns
    
    OUTPUT:
        database: Returns the database DataFrame with a new "Artifact" column
    """
    # Check if 'Time' exists
    if 'Time' not in database_wo_flats_artifacts.columns:
        raise KeyError("'Time' column is missing in database_wo_flats_artifacts")

    # Ensure 'Time' contains list-like values for explode()
    if not all(isinstance(x, (list, np.ndarray)) for x in database_wo_flats_artifacts['Time']):
        raise ValueError("'Time' must contain list-like values (e.g., lists or arrays)")

    # Explode 'Time'
    database_wo_flats_artifacts = database_wo_flats_artifacts.explode('Time')

    # Add a new column for labeled artifacts (0 = clean, 1 = artifact)
    database['Artifact'] = 0

    # Match artifacts from segmented data back to original data
    artifact_times = database_wo_flats_artifacts['Time']
    artifact_labels = database_wo_flats_artifacts['Artifact']

    # Use .loc to assign artifact labels where times match
    database.loc[database.Time.isin(artifact_times), 'Artifact'] = artifact_labels.values

    # Label EDA < 0.05 as artifact
    database.loc[database.Data < 0.05, 'Artifact'] = 1

    # print("7. Preparing final database with labeled artifacts completed")
    
    return database

def compute_eda_artifacts(data, window_size=60, SAMPLE_RATE=4):
    try:
        database = pd.DataFrame({'Data': data})

        # Generate timestamps if missing
        start_time = pd.Timestamp.now().floor('S')  # Start at current second
        interval = pd.Timedelta(milliseconds=int(1 / SAMPLE_RATE * 1000))  # 250ms interval
        database['Time'] = [start_time + i * interval for i in range(len(database))]

        # Fixed data length adjustment (prevents empty dataframe)
        remainder = len(database) % window_size
        if remainder != 0:
            database = database[:-remainder]

        database_copy = database.copy()

        # 1. Preprocess the EDA signals
        database_preprocessed = preprocess_eda_signals(database_copy)

        # 2. Segment EDA data
        database_segmented = segment_eda_data(database_preprocessed, window_size=60)

        # 3.1 Compute statistical and wavelet features
        database_features = compute_statistical_wavelet(database_segmented)

        # 3.2 Compute peaks features
        database_features = compute_peaks_features(database_features, SAMPLE_RATE=SAMPLE_RATE)

        # 4. Remove flat responses
        database_wo_flats = remove_flat_responses(database_features)

        features = ['EDA_median', 'EDA_mean', 'EDA_std', 'EDA_slope', 'EDA_min', 'EDA_max', 'EDA_fdmean', 'EDA_fdstd',
                    'EDA_sdmean', 'EDA_sdstd', 'EDA_drange', 'EDA_Phasic_median', 'EDA_Phasic_mean', 'EDA_Phasic_std',
                    'EDA_Phasic_slope', 'EDA_Phasic_min', 'EDA_Phasic_max', 'EDA_Phasic_fdmean', 'EDA_Phasic_fdstd',
                    'EDA_Phasic_sdmean', 'EDA_Phasic_sdstd', 'EDA_Phasic_drange', 'peaks_p', 'rise_time_p', 'amp_p',
                    'decay_time_p', 'SCR_width_p', 'auc_p', 'Wavelet3_mean', 'Wavelet3_median', 'Wavelet3_std',
                    'Wavelet2_mean', 'Wavelet2_median', 'Wavelet2_std', 'Wavelet1_mean', 'Wavelet1_median',
                    'Wavelet1_std']

        # 5. Identify artifacts in EDA
        database_wo_flats_artifacts = predict_shape_artifacts(features, database_wo_flats)

        # 6. Prepare final database with labeled artifacts
        database_w_artifacts = label_artifacts(database_wo_flats_artifacts, database)

        artifact_count = len(database['Artifact'].tolist()) - database['Artifact'].tolist().count(0)
        return artifact_count

    except ArithmeticError as e:
        print(f"Error encountered: {e}")
        return -1





# Keep all your existing functions as they are...

# Add this new function to process a single row in parallel

# Updated main function with parallelization

  

def main(data_file, window_size, out_path, SAMPLE_RATE):
    df = pd.read_csv(data_file)
    ans = []
    artifact_df = pd.DataFrame(columns=["PID", "arousal_category", "valence_category", "Artifact (%)"])

    for index, row in df.iterrows():
        print(index)
        data = row['Data']
        parts = data.split() 
        pid = row['PID']
        ac = row['arousal_category']
        vc = row['valence_category']
        data = [float(part.strip("[], ")) for part in parts if part.strip("[], ")]
        count_artifact = compute_eda_artifacts(data, window_size = window_size, SAMPLE_RATE = SAMPLE_RATE)
        if count_artifact == -1:
            ans.append(-1)
            artifact = "Not Applicable (not enough data)"
            
        else:
            percent_artifact = count_artifact/len(data)*100
            ans.append(percent_artifact)
            artifact = percent_artifact
            
        new_row = {"PID": pid, "arousal_category": ac, "valence_category": vc, "Artifact (%)": artifact}
        new_row_df = pd.DataFrame([new_row])
        artifact_df = pd.concat([artifact_df, new_row_df], ignore_index=True)
        # break

    # compute_eda_artifacts(database=df, out_path  = out_path,  window_size = window_size, SAMPLE_RATE = SAMPLE_RATE)
    
    ans = [x for x in ans if x != -1]
    if len(ans) == 0:
        average_artifact = 0
        sd_artifact = 0
    else:
        average_artifact = sum(ans)/len(ans)
        sd_artifact = np.std(ans)
        
    artifact_df.to_csv(out_path,index=False)
        
    return average_artifact, sd_artifact



    
    
