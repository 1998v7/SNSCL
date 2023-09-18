# SNSCL

PyTorch Implementation of SNSCL (accepted to CVPR 2023). 

### 1. Environment settings
Python 3.8,  Pytorch 1.11,  CUDA 11.1

### 2. Dataset
Before running the code, you should create a fold '/dataset/{}' and download datasets from following links. 

&nbsp;&nbsp; Stanford Dog: http://vision.stanford.edu/aditya86/ImageNetDogs/  
&nbsp;&nbsp; Stanford Car: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset  
&nbsp;&nbsp; Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/  
&nbsp;&nbsp; Cub-200-2011: https://data.caltech.edu/records/65de6-vp158  

### 3. Training
```
python vanilla_w_SNSCL.py  --dataset {dog, car, aircraft, cub}
                           --loss {ce_loss, APL, Asym, GCE, Sym, label_smooth, confPenalty}
                           --batch_size 64
                           --lr 0.002
                           --noise_type sym
                           --noise_r 0.2
                           --gpu 0
```

### Contact
If you have any problem about our code, feel free to contact 1998v7@gmail.com

### Cite
If you find the code useful, please consider citing our paper:
```
@inproceedings{wei2023fine,
  title={Fine-grained classification with noisy labels},
  author={Wei, Qi and Feng, Lei and Sun, Haoliang and Wang, Ren and Guo, Chenhui and Yin, Yilong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11651--11660},
  year={2023}
}
```


