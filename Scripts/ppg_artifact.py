import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from model import Model

def find_ppg_artifact(input_path, output_path, window_size = 60):
    # ---------------- Setup Device and Load Model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = 'TinyPPG_model_best_params.pth'

    # Instantiate your model and load state_dict
    model = Model()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    csv_path = input_path
    df = pd.read_csv(csv_path)

    results = []

    for idx, row in df.iterrows():
        pid = row.get('PID', None)
        ar = row.get('arousal_category', None)
        va = row.get('valence_category', None)
        eda_bs = row.get('Data', None)  # Adjust column name if needed
        if eda_bs is None:
            print(f"Row {idx} missing 'Data' for PID {pid}")
            continue

        # Ensure the data is a string
        if not isinstance(eda_bs, str):
            eda_bs = str(eda_bs)
        
        try:
            # Remove surrounding whitespace and brackets if present, then split by comma
            eda_bs = eda_bs.strip().strip("[]")
            parts = eda_bs.split(',')
            data_list = [float(x.strip()) for x in parts if x.strip()]
        except Exception as e:
            print(f"Error parsing data for PID {pid} in row {idx}: {e}")
            continue

        # print(f"Row {idx} (PID: {pid}) loaded successfully with {len(data_list)} data points.")

        # Reshape to [batch_size, channels, length] = [1, 1, length]
        try:
            data_array = np.array(data_list).reshape(1, 1, -1)
        except Exception as e:
            print(f"Error reshaping data for PID {pid}: {e}")
            continue
        data_tensor = torch.FloatTensor(data_array).to(device)
        # print (len(data_tensor[0][0]))
        # print (len(data_array[0][0]))
        window_list = list(torch.split(data_tensor, window_size, dim=2))
        # print(len(result_list))
        # print(result_list)
        # break

        # break

        # ---------------- Model Inference ----------------
        count = 0
        for temp_i in window_list:
            with torch.no_grad():
                out = model(data_tensor)
                # Assuming your model returns a dictionary with key 'seg'
                seg_output = out['seg']
                # Apply sigmoid if not already applied in the model
                seg_output = torch.sigmoid(seg_output)
                # Convert output to binary segmentation mask
                seg_output = (seg_output > 0.5).float()  # Expected shape: [1, 1, L]

                # ---------------- Aggregate to a Single Prediction ----------------
                # Option 1: Mean aggregation (e.g., average probability over sequence)
                prediction_mean = seg_output.mean().item()
                single_pred = 1 if prediction_mean > 0.3 else 0
                
                # Option 2: Majority vote aggregation (uncomment to use instead)
                # seg_array = seg_output.cpu().numpy().flatten()
                # single_pred = 1 if np.sum(seg_array) > (len(seg_array) / 2) else 0
                
                if single_pred == 1:
                    count += 1
                

                # pred_str = str(single_pred)
                # print(f"PID: {pid} - Prediction: {pred_str}")
                # print(pid, count, single_pred, prediction_mean)

        results.append({'PID': pid, 'valence_category': va, 'arousal_category': ar, 'Artifact (%)': ((count/len(window_list))*100)})

    # ---------------- Save the Predictions ----------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")

