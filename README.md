# ST-GAT
ST-GAT: Spatio-Temporal Graph Attention Network for TrafficFlow Prediction


![image](https://user-images.githubusercontent.com/92875660/138129249-05ff06a2-a949-4957-a45a-2a1dfed952ae.png)

This is an implementation of ST-GAT: Spatio-Temporal Graph Attention Network for TrafficFlow Prediction.


# Datasets
The processed datasets are available at Google Drive.
* https://drive.google.com/drive/folders/13uLQ14vehR0ztrlbNwXelE7fioejuEcb?usp=sharing

All datasets should be in './data'.


# Environment
* python 3.8.10
* pytorch 1.9.0
* numpy 1.19.5


# Commands

    # Train
    python ST-GAT.py --data=METR-LA --config=tmp
  
    # Test
    python ST-GAT.py --data=METR-LA --saved_model=tmp --config=tmp
