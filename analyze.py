import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelSpreading

def main():
    # Load RFM features
    print("Loading RFM features from sql_rfm_features.csv...")
    rfm_df = pd.read_csv('sql_rfm_features.csv')
    print(f"Loaded {rfm_df.shape[0]} customers.")

    # Features for clustering
    X = rfm_df[['recency', 'frequency', 'monetary']].values

    # Create labels based on Recency
    # Label 0: Active (Recency <= 30)
    # Label 1: Churned (Recency > 365)
    # Label -1: Unlabeled (otherwise)
    labels = np.full(rfm_df.shape[0], -1)  # default unlabeled
    active_mask = rfm_df['recency'] <= 30
    churned_mask = rfm_df['recency'] > 365
    labels[active_mask] = 0
    labels[churned_mask] = 1

    print(f"Number of active customers (label 0): {np.sum(labels == 0)}")
    print(f"Number of churned customers (label 1): {np.sum(labels == 1)}")
    print(f"Number of unlabeled customers (label -1): {np.sum(labels == -1)}")

    # Use LabelSpreading with KNN kernel
    print("Applying LabelSpreading with KNN kernel...")
    label_spread = LabelSpreading(kernel='knn', n_neighbors=7, max_iter=1000)
    label_spread.fit(X, labels)

    # Get the transduced labels (predicted labels for all data)
    predicted_labels = label_spread.transduction_

    # Add predictions to dataframe
    rfm_df['predicted_churn'] = predicted_labels

    # Save the updated dataframe with predictions (optional)
    rfm_df.to_csv('rfm_with_predictions.csv', index=False)
    print("Saved RFM with predictions to rfm_with_predictions.csv")

    # Visualization: Recency vs Monetary, color by predicted label
    plt.figure(figsize=(10, 6))
    # Define colors for each class: 0 (Active) -> green, 1 (Churned) -> red
    # We'll map predicted labels to colors
    colors = np.where(predicted_labels == 0, 'green', 'red')
    scatter = plt.scatter(rfm_df['recency'], rfm_df['monetary'], c=colors, alpha=0.6, edgecolors='k', linewidths=0.5)
    plt.xlabel('Recency (days since last order)')
    plt.ylabel('Monetary (total order value)')
    plt.title('Customer Churn Prediction: Recency vs Monetary')
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Active (Predicted)', markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Churned (Predicted)', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('churn_scatter.png', dpi=150)
    print("Saved scatter plot to churn_scatter.png")
    plt.close()

if __name__ == "__main__":
    main()