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
### Conv2d Block:
```python
self.layer_1    = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2,bias=True),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(16)
        )
```  
### Linear Block:
```python
self.linear_1   = torch.nn.Sequential(
            torch.nn.Linear(256*2*2,256),
            torch.nn.Tanh()
        )
```
## The Input:    
**Shape: torch.Tensor->(3,512,512) float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/BaseImg.jpg?raw=true)

## The Intermediate: 

### Layer1, Ch1-16: Epoch 9
Shown here are the transformations done to the image by each of the 16 channels in Layer1. Yellow represents the most negative output, and Blue represents the most positive output.  

**Shape (per Ch): torch.Tensor->(1,512,512) dtype=torch.float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ep0.jpg?raw=true)


### Layer1, Ch1: Epoch 1-9
Transformation done to the image by Ch1 in **Layer1** over 9 epochs of training.  

**Shape (per Ch): torch.Tensor->(1,512,512) dtype=torch.float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ch1.jpg?raw=true)


### Layer2, Ch1: Epoch 1-9
Transformation done to the image by Ch1 in **Layer2** over 9 epochs of training. Note how Layer2 changes more in training than Layer1  

**Shape (per Ch): torch.Tensor->(1,256,256) dtype=torch.float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer2_ch1.jpg?raw=true)


### Layer3, Ch1: Epoch 1-9
Transformation done to the image by Ch1 in **Layer3** over 9 epochs of training. Very pronounced changes

**Shape (per Ch): torch.Tensor->(1,128,128) dtype=torch.float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer3_ch1.jpg?raw=true)


### Last Layer, Ch1: Epoch 1-9
Transformation done to the image by Ch1 in **Layer8** over 9 epochs of training. Huge changes

**Shape (per Ch): torch.Tensor->(1,2,2) dtype=torch.float32**  
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer8_ch1.jpg?raw=true)

## The Output:

- Dawnbreaker:0%
- Floodzone: 93.6%
- Silk Roead: 0%
- Lost Islands: 6.3%
- Siege of Shanghai: 0%
- **Zavod 311: .1%**