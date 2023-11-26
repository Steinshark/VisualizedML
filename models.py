import torch 

class basicNet(torch.nn.Module):

    def __init__(self,n_classes):

        super(basicNet,self).__init__() 

        self.conv_activation    = torch.nn.functional.relu
        self.lin_activation     = torch.nn.functional.leaky_relu 

        self.conv1      = torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.conv2      = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.conv3      = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,stride=1,padding=2)
        self.avg_pool1  = torch.nn.AdaptiveAvgPool2d(256)

        self.conv4      = torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.avg_pool2  = torch.nn.AdaptiveAvgPool2d(128)

        self.conv5      = torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2)
        self.avg_pool3  = torch.nn.AdaptiveAvgPool2d(64)

        self.conv6      = torch.nn.Conv2d(in_channels=256,out_channels=128,kernel_size=5,stride=1,padding=2)
        self.avg_pool4  = torch.nn.AdaptiveAvgPool2d(32)

        self.conv7      = torch.nn.Conv2d(in_channels=128,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.avg_pool5  = torch.nn.AdaptiveAvgPool2d(16)

        self.conv8      = torch.nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.avg_pool6  = torch.nn.AdaptiveAvgPool2d(8)

        self.lin1       = torch.nn.Linear(32*8*8,512)
        self.drop1      = torch.nn.Dropout(.25)

        self.lin2       = torch.nn.Linear(512,64)
        self.drop2      = torch.nn.Dropout(.1)

        self.lin3       = torch.nn.Linear(64,n_classes)

    
    def forward(self,x:torch.Tensor)->torch.Tensor:

        x               = self.conv_activation(self.conv1(x))
        x               = self.conv_activation(self.conv2(x))
        x               = self.conv_activation(self.conv3(x))
        x               = self.avg_pool1(x)

        x               = self.conv_activation(self.conv4(x))
        x               = self.avg_pool2(x)

        x               = self.conv_activation(self.conv5(x))
        x               = self.avg_pool3(x)

        x               = self.conv_activation(self.conv6(x))
        x               = self.avg_pool4(x)

        x               = self.conv_activation(self.conv7(x))
        x               = self.avg_pool5(x)

        x               = self.conv_activation(self.conv8(x))
        x               = self.avg_pool6(x)
        x               = x.view(x.shape[0],-1)
        x               = self.lin_activation(self.lin1(x))
        x               = self.drop1(x)

        x               = self.lin_activation(self.lin2(x))
        x               = self.drop2(x)

        x               = torch.nn.functional.sigmoid(self.lin3(x))


        return x 

