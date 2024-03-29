import torch 

class basicNet(torch.nn.Module):

    def __init__(self,n_classes):

        super(basicNet,self).__init__() 

        self.conv_activation    = torch.nn.functional.leaky_relu
        self.lin_activation     = torch.nn.functional.leaky_relu 

        k_size          = 5 
        self.conv1      = torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=True)
        #self.conv1_5    = torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn1        = torch.nn.BatchNorm2d(16)
        self.avg_pool1  = torch.nn.MaxPool2d(2)

        self.conv2      = torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn2        = torch.nn.BatchNorm2d(32)
        self.avg_pool2  = torch.nn.MaxPool2d(2)

        self.conv3      = torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn3        = torch.nn.BatchNorm2d(64)
        self.avg_pool3  = torch.nn.MaxPool2d(2)

        self.conv4      = torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn4        = torch.nn.BatchNorm2d(64)
        self.avg_pool4  = torch.nn.MaxPool2d(2)

        self.conv5      = torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn5        = torch.nn.BatchNorm2d(128)
        self.avg_pool5  = torch.nn.MaxPool2d(2)

        self.conv6      = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn6        = torch.nn.BatchNorm2d(128)
        self.avg_pool6  = torch.nn.MaxPool2d(2)

        self.conv7      = torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False)
        self.bn7        = torch.nn.BatchNorm2d(128)
        self.avg_pool7  = torch.nn.MaxPool2d(2)

        self.lin1       = torch.nn.Linear(128*4*4,512)

        self.lin2       = torch.nn.Linear(512,n_classes)

    
    def forward(self,x:torch.Tensor)->torch.Tensor:

        x               = self.conv_activation(self.conv1(x))
        #x               = self.conv_activation(self.conv1_5(x))
        x               = self.bn1(x)
        x               = self.avg_pool1(x)

        x               = self.conv_activation(self.conv2(x))
        x               = self.bn2(x)
        x               = self.avg_pool2(x)

        x               = self.conv_activation(self.conv3(x))
        x               = self.bn3(x)
        x               = self.avg_pool3(x)

        x               = self.conv_activation(self.conv4(x))
        x               = self.bn4(x)
        x               = self.avg_pool4(x)

        x               = self.conv_activation(self.conv5(x))
        x               = self.bn5(x)
        x               = self.avg_pool5(x)

        x               = self.conv_activation(self.conv6(x))
        x               = self.bn6(x)
        x               = self.avg_pool6(x)

        x               = self.conv_activation(self.conv7(x))
        x               = self.bn7(x)
        x               = self.avg_pool7(x)


        x               = x.view(x.shape[0],-1)
        x               = self.lin_activation(self.lin1(x))
        #x               = self.drop1(x)

        x               = torch.nn.functional.softmax(self.lin2(x),dim=1)
        #x               = self.drop2(x)

        #x               = self.lin3(x)


        return x 


    def view_ch(self,img:torch.tensor,layer:int,ch:int):
        
        #Pass image through to layer 'layer'

        if layer == 1:
            pass



class visNet(torch.nn.Module):

    def __init__(self,n_classes,k_size=5,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

        super(visNet,self).__init__() 

        self.conv_activation    = torch.nn.Tanh
        self.lin_activation     = torch.nn.LeakyReLU
        self.device             = device

        self.layer_1    = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3,out_channels=16,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(16)
        )

        self.layer_2    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(32)
        )
        
        self.layer_3    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32,out_channels=32,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(32)
        )

        self.layer_4    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(64)
        )
        
        self.layer_5    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64,out_channels=64,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(64)
        )
        
        self.layer_6    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(128)
        )

        self.layer_7    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(128)
        )

        self.layer_8    = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(in_channels=128,out_channels=256,kernel_size=k_size,stride=1,padding=int(k_size/2),bias=False),
            self.conv_activation(),
            torch.nn.BatchNorm2d(256)
        )

        self.linear_1   = torch.nn.Sequential(
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),

            torch.nn.Linear(256*2*2,256),
            self.lin_activation(),
            torch.nn.Linear(256,n_classes),
            torch.nn.Softmax(dim=1)
        )

        self.layers     = [self.layer_1,self.layer_2,self.layer_3,self.layer_4,self.layer_5,self.layer_6,self.layer_7,self.layer_8]
        self.to(self.device)
    

    def forward(self,x:torch.Tensor)->torch.Tensor:

        x               = self.layer_1(x)
        x               = self.layer_2(x)
        x               = self.layer_3(x)
        x               = self.layer_4(x)
        x               = self.layer_5(x)
        x               = self.layer_6(x)
        x               = self.layer_7(x)
        x               = self.layer_8(x)

        x               = self.linear_1(x)
        return x 


    def view_layer(self,img:torch.Tensor,layer:int,negative_color=(255,209,41),positive_color=(77,115,255)):
        
        if layer == 0:
            return [img.type(torch.float32).to(self.device) / torch.max(img)]
        
        with torch.no_grad():
            
            #Correct shape  
            if not len(img.shape) == 4:
                img                 = img.unsqueeze(0)
            #Forward pass to correct layer 
            forward_passed          = img.type(torch.float32).to(self.device)
            for l in range(layer):
                forward_passed      = self.layers[l](forward_passed)

            forward_passed          = forward_passed[0]
            
            #Prep to store final channel outputs
            colorized_channels  = [] 

            #Process
            for channel in forward_passed:

                #Separate neg and pos vals. Put both in positive alues
                negative_values     = torch.where(channel > 0, torch.tensor(0),channel) * -1
                positive_values     = torch.where(channel < 0, torch.tensor(0),channel)

                #Scale to 1 
                negative_values_max = torch.max(negative_values) if torch.max(negative_values) > .0001 else 1 
                positive_values_max = torch.max(positive_values) if torch.max(positive_values) > .0001 else 1
                negative_values     /= negative_values_max
                positive_values     /= positive_values_max 

                #Create 3 channel
                negative_values     = torch.stack([torch.clone(negative_values),torch.clone(negative_values),torch.clone(negative_values)])
                positive_values     = torch.stack([torch.clone(positive_values),torch.clone(positive_values),torch.clone(positive_values)])

                #Add color
                for ch in range(3):
                    negative_values[ch]     *= negative_color[ch]
                    positive_values[ch]     *= positive_color[ch]

                #Combine 
                colorized_img       = negative_values + positive_values

                #Scale back down to 1 
                colorized_img       /= 255

                #Add to deck
                colorized_channels.append(colorized_img)

            #Return
            return colorized_channels 
            



                





            