import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))

    # Train PR curve
    plt.figure(figsize=(8,6))
    plt.title('Precision-Recall Curve, Train')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for i, file_name in enumerate(["train_preds1.csv", "train_preds2.csv"]):
        f = os.path.join(cwd, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        precision, recall, _ = precision_recall_curve(results["y_true"], results["y_probs_pos"], pos_label=1)
        auc_score = auc(recall, precision)

        plt.plot(recall, precision, label=f'Model {i + 1} (AUC = {auc_score:.2f})')

    plt.legend()
    plt.savefig(os.path.join(cwd, "PR_train.png"))

    # Val PR curve
    plt.figure(figsize=(8,6))
    plt.title('Precision-Recall Curve, Val')
    plt.xlabel('Recall')
    plt.ylabel('Precision')

    for i, file_name in enumerate(["val_preds1.csv", "val_preds2.csv"]):
        f = os.path.join(cwd, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        precision, recall, _ = precision_recall_curve(results["y_true"], results["y_probs_pos"], pos_label=1)
        auc_score = auc(recall, precision)

        plt.plot(recall, precision, label=f'Model {i + 1} (AUC = {auc_score:.2f})')

    plt.legend()
    plt.savefig(os.path.join(cwd, "PR_val.png"))