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
The model consists of 7 blocks of convolution layers, and 2 blocks of linear layers
Block:
```python
self.layer_1    = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2,bias=True),
            torch.nn.LeakyRelu(),
            torch.nn.BatchNorm2d(16)
        )
```
## The Input:    
**Shape: torch.Tensor->(3,512,512) float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/BaseImg.jpg?raw=true)

## The Output: 
Shown are the transformations done to the image by each of the 16 channels in Layer1. Red represents the most negative output, and Blue represents the most positive output.  

**Shape: torch.Tensor->(16,512,512) float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ep0.jpg?raw=true)


Below is the transformation done to the image by Ch1 in Layer1 through 9 epochs of training.  

**Shape: torch.Tensor->(1,512,512) float32(())**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ch1.jpg?raw=true)
