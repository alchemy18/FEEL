from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_squared_error, r2_score, classification_report
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import warnings
warnings.filterwarnings('ignore')
random_seed = 42

def create_unified_emotion_label(arousal, valence):
    """
    Create unified emotion label based on arousal and valence values:
    - 0: arousal=1, valence=0
    - 1: arousal=1, valence=1  
    - 2: arousal=0, valence=0
    - 3: arousal=0, valence=1
    """
    if arousal == 1 and valence == 0:
        return 0
    elif arousal == 1 and valence == 1:
        return 1
    elif arousal == 0 and valence == 0:
        return 2
    elif arousal == 0 and valence == 1:
        return 3
    else:
        raise ValueError(f"Invalid arousal/valence combination: arousal={arousal}, valence={valence}")

def train_MLP_unified_emotion(X_train, y_train_emotion, X_test, y_test_emotion, hidden_layer_sizes=(100,), random_seed=42):
    """Train MLP model for unified emotion classification (4 classes)"""
    
    # Create and fit the MLP model for unified emotion
    model_emotion_mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_seed)
    model_emotion_mlp.fit(X_train, y_train_emotion)

    # Predict on the test set
    y_pred_emotion_mlp = model_emotion_mlp.predict(X_test)

    # Calculate metrics
    accuracy_emotion_mlp = accuracy_score(y_test_emotion, y_pred_emotion_mlp)
    balanced_acc_emotion_mlp = balanced_accuracy_score(y_test_emotion, y_pred_emotion_mlp)
    
    # For multiclass F1 score, use 'weighted' average
    f1_emotion = f1_score(y_test_emotion, y_pred_emotion_mlp, average='weighted')
    
    # R2 and MSE for multiclass
    r2_emotion = r2_score(y_test_emotion, y_pred_emotion_mlp)
    mse_emotion = mean_squared_error(y_test_emotion, y_pred_emotion_mlp)

    return accuracy_emotion_mlp, balanced_acc_emotion_mlp, r2_emotion, mse_emotion, f1_emotion

def UnifiedEmotion(PathFile, output_file=None):
    """
    Train MLP for unified emotion classification combining arousal and valence
    """
    if not(PathFile[-16:]=="Features_EDA.csv" or PathFile[-16:]=="Features_PPG.csv" or PathFile[-21:]=="Features_Combined.csv"):
        raise AssertionError(f"Path file is not valid, please check {PathFile[-16:]}")
    
    if PathFile[-16:] == "Features_EDA.csv":
        signal_type = "EDA"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR']
    elif PathFile[-16:] == "Features_PPG.csv":
        signal_type = "PPG"
        relevent_features = ['BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
    else:
        signal_type = "Combined"
        relevent_features = ['ku_eda','sk_eda','dynrange','slope','variance','entropy','insc','fd_mean','max_scr','min_scr','nSCR','meanAmpSCR','meanRespSCR','sumAmpSCR','sumRespSCR','BPM', 'PPG_Rate_Mean', 'HRV_MedianNN',
       'HRV_Prc20NN', 'HRV_MinNN', 'HRV_HTI', 'HRV_TINN', 'HRV_LF', 'HRV_VHF',
       'HRV_LFn', 'HRV_HFn', 'HRV_LnHF', 'HRV_SD1SD2', 'HRV_CVI', 'HRV_PSS',
       'HRV_PAS', 'HRV_PI', 'HRV_C1d', 'HRV_C1a', 'HRV_DFA_alpha1',
       'HRV_MFDFA_alpha1_Width', 'HRV_MFDFA_alpha1_Peak',
       'HRV_MFDFA_alpha1_Mean', 'HRV_MFDFA_alpha1_Max',
       'HRV_MFDFA_alpha1_Delta', 'HRV_MFDFA_alpha1_Asymmetry', 'HRV_ApEn',
       'HRV_ShanEn', 'HRV_FuzzyEn', 'HRV_MSEn', 'HRV_CMSEn', 'HRV_RCMSEn',
       'HRV_CD', 'HRV_HFD', 'HRV_KFD', 'HRV_LZC']
        
    X_train = pd.read_csv(PathFile)
    X_train = X_train.fillna(0)
    X_train.replace([np.inf, -np.inf], 0, inplace=True)
    
    # Create unified emotion labels
    X_train['unified_emotion'] = X_train.apply(
        lambda row: create_unified_emotion_label(row['arousal_category'], row['valence_category']), 
        axis=1
    )
    
    results = []
    accuracy_emotion_mlp, balanced_acc_emotion_mlp, r2_emotion, mse_emotion, f1_emotion = {}, {}, {}, {}, {}
    
    for pi in tqdm(X_train['PID'].unique(), desc='Processing PID'):
        test_data = X_train[X_train['PID']==pi]
        train_data = X_train[X_train['PID']!=pi]

        train_x = train_data[relevent_features]
        train_emotion = train_data["unified_emotion"]

        test_x = test_data[relevent_features]
        test_emotion = test_data["unified_emotion"]

        # Combine features and labels for balancing
        train_data_combined = pd.concat([train_x, train_emotion], axis=1)
        
        # Get class counts
        class_counts = train_data_combined[train_emotion.name].value_counts()
        max_count = class_counts.max()
        
        # Balance all classes by oversampling to match the majority class
        balanced_data = []
        
        for class_label in class_counts.index:
            class_data = train_data_combined[train_data_combined[train_emotion.name] == class_label]
            
            if len(class_data) < max_count:
                # Oversample minority classes
                class_oversampled = resample(class_data, 
                                           replace=True,
                                           n_samples=max_count,
                                           random_state=42)
                balanced_data.append(class_oversampled)
            else:
                balanced_data.append(class_data)
        
        # Combine all balanced classes
        train_data_balanced = pd.concat(balanced_data, ignore_index=True)
        
        # Separate features and labels after balancing
        train_x = train_data_balanced.drop(columns=[train_emotion.name])
        train_emotion = train_data_balanced[train_emotion.name]
        
        # Train model
        accuracy_emotion_mlp[pi], balanced_acc_emotion_mlp[pi], r2_emotion[pi], mse_emotion[pi], f1_emotion[pi] = train_MLP_unified_emotion(
            train_x, train_emotion, test_x, test_emotion
        )
        
        results.append({
            'PID': pi,
            'Signal Type': signal_type,
            'Input Type': 'Feature',
            'Model Name': 'MLP',
            'Classification': 'Unified_Emotion',
            'Test_Accuracy': accuracy_emotion_mlp[pi],
            'Test_F1': f1_emotion[pi]
        })
    
    # Calculate average metrics
    acc_emotion = sum(accuracy_emotion_mlp.values()) / len(accuracy_emotion_mlp.values())
    f1_emotion_avg = sum(f1_emotion.values()) / len(f1_emotion.values())
    
    print(f"------ Average accuracy of MLP on unified emotion classification: {acc_emotion:.4f} ------")
    print(f"------ Average F1 score of MLP on unified emotion classification: {f1_emotion_avg:.4f} ------")
    
    # Print class distribution for debugging
    print("Unified emotion class distribution:")
    print("0 (A=1,V=0):", (X_train['unified_emotion'] == 0).sum())
    print("1 (A=1,V=1):", (X_train['unified_emotion'] == 1).sum())
    print("2 (A=0,V=0):", (X_train['unified_emotion'] == 2).sum())
    print("3 (A=0,V=1):", (X_train['unified_emotion'] == 3).sum())
    
    if output_file:
        # Convert the results list into a DataFrame
        df_results = pd.DataFrame(results)
        
        if os.path.exists(output_file):
            # Read the existing CSV into a DataFrame
            existing_df = pd.read_csv(output_file)
            # Append the new results to the existing DataFrame
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        
        # Write the combined DataFrame to CSV
        df_results.to_csv(output_file, index=False)
        
    return acc_emotion, f1_emotion_avg
