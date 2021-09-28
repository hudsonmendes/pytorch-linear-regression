import torch
import numpy as np
import pandas as pd
from ml import LogisticRegressionModel, LogisticRegressionTrainer

if __name__ == '__main__':
    model = LogisticRegressionModel(input_dim=1, output_dim=1)

    dataset = pd.read_csv("./data/sample.csv")
    
    trainer = LogisticRegressionTrainer(model, 0.00000005)
    trainer.train(dataset, 100)

    with torch.no_grad():
        while True:
            try:
                print("What year?", end=' ')
                x = int(input())
                if x:
                    x = torch.autograd.Variable(torch.from_numpy(np.array([x], dtype=np.float32)))
                    prediction = model(x).data.numpy()
                    print(f">>> {prediction}")
                else:
                    break
            except KeyboardInterrupt or EOFError:
                print("Closed by User...")
                break