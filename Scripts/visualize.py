import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

def load_data(file_path):
    df = pd.read_csv(file_path)
    feature_columns = [col for col in df.columns if col not in ['PID','arousal_category','valence_category']]
    X = df[feature_columns].values
    y_arousal = df['arousal_category'].values
    y_valence = df['valence_category'].values
    return X, y_arousal, y_valence


def apply_umap(X):
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    return umap_reducer.fit_transform(X)

def apply_tsne(X):
    tsne_reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne_reducer.fit_transform(X)

def plot_embedding(X_embedded, labels, output_path):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Label')
    t = output_path[-16:-4]
    # print(t)
    plt.title(t)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return None

def visualize(input_path, output_folder):
    try:
        X, y_arousal, y_valence = load_data(file_path = input_path)
        
        X_umap = apply_umap(X)
        X_tsne = apply_tsne(X)
        
        plot_embedding(X_umap, y_arousal, output_folder + "UMAP_Arousal.png")
        plot_embedding(X_umap, y_valence, output_folder + "UMAP_Valence.png")
        plot_embedding(X_tsne, y_arousal, output_folder + "tSNE_Arousal.png")
        plot_embedding(X_tsne, y_valence, output_folder + "tSNE_Valence.png")
        
        return ("Plots Saved")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

# Usage:
# For regular plotting:
# process_and_plot('your_data_file.csv')

# For participant-wise plotting:
# plot_participant_wise('your_data_file.csv')