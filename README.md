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


# Experiments

Table: Training and inference time comparison.

![image](https://user-images.githubusercontent.com/92875660/171132693-74119049-db87-4508-a2c2-7a1dde13d846.png)


# Authors

Junho Song
Jiwon Son
Dong-hyuk Seo
Kyungsik Han
Namhyuk Kim
Sang-Wook Kim

# Cite
We encourage you to cite our paper if you have used the code in your work. You can use the following BibTex citation:
@inproceedings{song2022st,
  title={ST-GAT: A Spatio-Temporal Graph Attention Network for Accurate Traffic Speed Prediction},
  author={Song, Junho and Son, Jiwon and Seo, Dong-hyuk and Han, Kyungsik and Kim, Namhyuk and Kim, Sang-Wook},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={4500--4504},
  year={2022}
}
