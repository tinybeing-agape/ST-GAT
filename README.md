# ST-GAT


![image](https://user-images.githubusercontent.com/92875660/138129249-05ff06a2-a949-4957-a45a-2a1dfed952ae.png)

This is an implementation of Spatio-Temporal Graph Attention Network.


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
    python ST-GAT.py --mode=train --data=METR-LA --conf=./config/metr-la.conf
  
    # Test
    python ST-GAT.py --mode=test --data=METR-LA --saved_model=./out/metr-la/best_model --conf=./config/metr-la.conf
