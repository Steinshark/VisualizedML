Welcome to my project exploring visualization of Neural Networks!

The Problem: This model classifies screenshots of Battlefield4 by map. 

The Architecture: The model consists of 7 Conv2D layers of size 5 kernels, and 2 Linear layers. There are currently 6 BF4 maps: {Dawnbreaker,Floodzone,Silk Road,Siege of Shanghai,Lost Islands, Zavod 311}


Base Image: 
Shape: torch.Tensor->(3,512,512) float32
![alt text](https://github.com/steinshark/VisualizedML/main/BaseImg.jpg?raw=true)

Visualization of all 16 First Layer channel outputs
Shape: torch.Tensor->(16,512,512) float32
![alt text](https://github.com/steinshark/VisualizedML/blob/main/Layer1_ep0.jpg?raw=true)

Visualization of First Layer, First channel, epoch 0-9 
![alt text](https://github.com/steinshark/VisualizedML/main/blob/Layer1_ch1.jpg?raw=true)
