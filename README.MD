# Visualizing Neural Networks
This repo is a project of mine to visualize neural networks as they learn. I'm interested in tracking:
- [ ] The transformation of the image through each layer
- [ ] The kernels learned in each layer.

## The Problem: 
This PyTorch model classifies screenshots of Battlefield4 by map. 
Currently, the following maps are part of the dataset:
- Dawnbreaker
- Floodzone
- Silk Road
- Siege of Shanghai
- Lost Islands
- Zavod 311

## The Architecture: 
The model consists of 7 blocks of convolution layers, and 2 Linear layers
Block:
```python
self.layer_1    = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=True),
            self.conv_activation(),
            torch.nn.BatchNorm2d(16)
        )
```

Base Image:  
Shape: torch.Tensor->(3,512,512) float32
![alt text](https://github.com/steinshark/VisualizedML/blob/main/BaseImg.jpg?raw=true)

Visualization of the image after pass through each of the 16 Channels in Layer1 of the Network.  
Shape: torch.Tensor->(16,512,512) float32
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ep0.jpg?raw=true)

Visualization of the image after pass through the 1st Channel through each training epoch  
Shape: torch.Tensor->(1,512,512) float32
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ch1.jpg?raw=true)
