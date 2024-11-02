import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == "__main__":
    num_images = 10
    cwd = os.path.dirname(os.path.realpath(__file__))

    for i, file_name in enumerate(["train_preds1.csv", "train_preds2.csv"]):
        f = os.path.join(cwd, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        # Filter so that only non-schools are in dataframe
        results = results[results["y_true"] == 0]

        # Sort by positive class probs and take first N rows
        ns_worst_rows = results.sort_values(by="y_probs_pos", ascending=False).head(num_images).reset_index(drop=True)

        plt.figure(figsize=(16,10))
        for i, row in ns_worst_rows.iterrows():
            path = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
            image = Image.open(path).convert("RGB")

            plt.subplot(2, int(num_images / 2), i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.savefig(os.path.join(cwd, f"{num_images}_worst_ns_train_{i}.png"))

    for i, file_name in enumerate(["val_preds1.csv", "val_preds2.csv"]):
        f = os.path.join(cwd, file_name)
        results = pd.read_csv(f)

        # Set positive class probs
        results["y_probs_pos"] = results.apply(lambda row: row["y_probs"] if row["y_preds"] == 1 else 1 - row["y_probs"], axis=1)

        # Filter so that only non-schools are in dataframe
        results = results[results["y_true"] == 0]

        # Sort by positive class probs and take first N rows
        ns_worst_rows = results.sort_values(by="y_probs_pos", ascending=False).head(num_images).reset_index(drop=True)

        plt.figure(figsize=(16,10))
        for i, row in ns_worst_rows.iterrows():
            path = f"/mnt/sdb/agorup/school_mapping/satellite_images/large/VNM/non_school/{row['UID']}.jpeg"
            image = Image.open(path).convert("RGB")

            plt.subplot(2, int(num_images / 2), i + 1)
            plt.imshow(image)
            plt.axis('off')
        plt.savefig(os.path.join(cwd, f"{num_images}_worst_ns_val_{i}.png"))