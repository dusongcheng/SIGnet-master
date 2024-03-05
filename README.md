Progressive Spatial Information-Guided Deep Aggregation Convolutional Network for Hyperspectral Spectral Super-Resolution, TNNLS, 2023.
==
[Jiaojiao Li](https://scholar.google.com/citations?user=Ccu3-acAAAAJ&hl=zh-CN&oi=sra), [Songcheng Du](https://github.com/dusongcheng), [Rui song](https://scholar.google.com/citations?user=_SKooBYAAAAJ&hl=zh-CN), [Yunsong Li](https://dblp.uni-trier.de/pid/87/5840.html), and [Qian Du](https://scholar.google.com/citations?user=0OdKQoQAAAAJ&hl=zh-CN).

***
Code for the paper: [Progressive Spatial Information-Guided Deep Aggregation Convolutional Network for Hyperspectral Spectral Super-Resolution](https://ieeexplore.ieee.org/abstract/document/10298249).


<div align=center><img src="/Image/network.png" width="100%" height="100%"></div>
Fig. 1: Network architecture of our accurate SIGnet for ssr.

Training and Test Process
--
1) Please prepare the training and test data as operated in the paper. 
3) Run "train.py" to train the SIGnet.
4) Run "test.py" to test.
5) Download the pretrained model ([Baidu Disk](https://pan.baidu.com/s/1uDZ8IXFhlmpnBa-ujpA01g?pwd=abcd), code: `abcd`)).

References
--
If you find this code helpful, please kindly cite:

[1] Li J, Du S, Song R, et al. Progressive Spatial Information-Guided Deep Aggregation Convolutional Network for Hyperspectral Spectral Super-Resolution[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.

[2] J. Li, S. Du, R. Song, C. Wu, Y. Li and Q. Du, "HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution of RGB Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-15, 2022, Art no. 5522515, doi: 10.1109/TGRS.2022.3142258.

[3] Li J, Du S, Wu C, et al. DRCR Net: Dense Residual Channel Re-Calibration Network With Non-Local Purification for Spectral Super Resolution[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022: 1259-1268.

[4] Du S, Leng Y, Liang X, et al. Degradation Aware Unfolding Network for Spectral Super-Resolution[J]. IEEE Geoscience and Remote Sensing Letters, 2023.


Citation Details
--
BibTeX entry:
```
@article{li2023progressive,
  title={Progressive Spatial Information-Guided Deep Aggregation Convolutional Network for Hyperspectral Spectral Super-Resolution},
  author={Li, Jiaojiao and Du, Songcheng and Song, Rui and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}

@ARTICLE{9678983,
  author={Li, Jiaojiao and Du, Songcheng and Song, Rui and Wu, Chaoxiong and Li, Yunsong and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HASIC-Net: Hybrid Attentional Convolutional Neural Network With Structure Information Consistency for Spectral Super-Resolution of RGB Images}, 
  year={2022},
  volume={60},
  number={},
  pages={1-15},
  doi={10.1109/TGRS.2022.3142258}}

@InProceedings{Li_2022_CVPR,
    author    = {Li, Jiaojiao and Du, Songcheng and Wu, Chaoxiong and Leng, Yihong and Song, Rui and Li, Yunsong},
    title     = {DRCR Net: Dense Residual Channel Re-Calibration Network With Non-Local Purification for Spectral Super Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1259-1268}
}

@article{du2023degradation,
  title={Degradation Aware Unfolding Network for Spectral Super-Resolution},
  author={Du, Songcheng and Leng, Yihong and Liang, Xinyi and Li, Jiaojiao and Liu, Wei and Du, Qian},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={2023},
  publisher={IEEE}
}
```

Licensing
--
Copyright (C) 2023 Songcheng Du

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.
