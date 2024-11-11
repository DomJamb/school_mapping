import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cwd = os.getcwd()
    f1_model1_path = os.path.join(cwd, 'f1_model1.csv')
    f1_model2_path = os.path.join(cwd, 'f1_model2.csv')
    
    f1_model1 = pd.read_csv(f1_model1_path)
    f1_model2 = pd.read_csv(f1_model2_path)

    # Train F1 curve
    plt.figure(figsize=(8,6))
    plt.title('Train F1 over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 (%)')

    for i, f1_csv in enumerate([f1_model1, f1_model2]):
        plt.xticks(f1_csv['epoch'])
        plt.plot(f1_csv['epoch'], f1_csv['train'], label=f'{"North" if i == 0 else "South"}')

    plt.legend()
    plt.savefig(os.path.join(cwd, 'F1_train.png'))

    # Val F1 curve
    plt.figure(figsize=(8,6))
    plt.title('Val F1 over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 (%)')

    for i, f1_csv in enumerate([f1_model1, f1_model2]):
        plt.xticks(f1_csv['epoch'])
        plt.plot(f1_csv['epoch'], f1_csv['val'], label=f'{"North -> South" if i == 0 else "South -> North"}')

    plt.legend()
    plt.savefig(os.path.join(cwd, 'F1_val.png'))