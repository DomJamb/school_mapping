from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("oai_inference_max_pred.csv")
preds = df["pred"].values

plt.figure()
plt.title("Predictions for inference tiles with Anditi schools (min distance)")
plt.hist(preds, bins=20, range=(0,1))
#plt.show()

plt.savefig("max_pred.png")
