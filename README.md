# SGCN
Dataset and Code for the paper "SVGC-AVA: 360-Degree Video Saliency Prediction with Spherical Vector-Based Graph Convolution and Audio-Visual Attention", TMM

## Requirements
* Python3 == 3.8.5
* Pytorch == 1.7.1

## Datasets and Pretrained Model
You can download the dataset and models at https://drive.google.com/drive/folders/15WBe_AYPs1geSwft9UUA5x5fQzngXsJF?usp=drive_link

## Usages
1.Install the dependencies
~~~
pip install -r requirements.txt
~~~
2.Generate the dataset
~~~
python data_prepare_qin.py
python data_prepare_chao.py
~~~

3.Train and test the model
~~~
sh main.sh
~~~

## Citation
If you find this code and dataset is useful for your research, please cite our paper "SVGC-AVA: 360-Degree Video Saliency Prediction with Spherical Vector-Based Graph Convolution and Audio-Visual Attention"

~~~
~~~
