import torch 
import models 
from torch.utils.data import Dataset, DataLoader
import torchvision 
from matplotlib import pyplot as plt
import random 
import time 
import torchvision.utils
import numpy 

class imgDataSet(Dataset):

    def __init__(self,data_dict:dict,n_iters=10,load_n=8):

        self.data           = [] 
        self.to_pil_img     = torchvision.transforms.ToPILImage()

        self.labels         = list(set([d[0] for d in data_dict.values()]))

        for path in data_dict:

            for _ in range(n_iters):
                label                   = self.labels.index(data_dict[path][0])
                indices                 = random.choices(range(data_dict[path][1]),k=load_n)

                for i in indices:
                    tsr                     = torch.load(f"{path}{i}.tsr")
                    label_p                 = torch.zeros(3)
                    label_p[label]          = 1
                    self.data.append([tsr[random.randint(0,29)],label_p])
        


            # #Load the mp4 in chunks of 8

            # for i in range(5000):
                
            #     if i % 100 == 0:
            #         print(f"i={i}\n")
            #     video       = torchvision.io.read_video(data_dict[path],pts_unit='sec',start_pts=i,end_pts=i+1,output_format="TCHW")[0]
            #     video.requires_grad_(False)

            #     if video.shape[0] == 1:
            #         break 
            #     else:

            #         #Cut to 512x512
            #         dims        = 512
            #         h           = video.shape[2]
            #         w           = video.shape[3]
            #         w_start     = int(int(w/2) - int(dims/2))
            #         h_start     = int(int(h/2) - int(dims/2))


            #         video       = video[:,:,h_start:h_start+dims,w_start:w_start+dims]
            #         torch.save(video,data_dict[path].replace(".mp4",f"/{i}.tsr"))
            #         #print(f"\t{video.shape}")  

            


            
    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        return self.data[i][0],self.data[i][1]

class Trainer:

    def __init__(self):
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.epoch      = 0 
        self.loss_fn    = torch.nn.CrossEntropyLoss()
        self.losses     = {}    
        self.to_pil_img = torchvision.transforms.ToPILImage()


    def select_model(self,lr=.0002):
        self.model      = models.basicNet(n_classes=3).to(self.device)
        self.optimizer  = torch.optim.Adam(self.model.parameters(),lr=lr)


    def select_dataset(self,n_iters=32,load_n=4):
        self.dataset    = imgDataSet({
                                        "C:/users/Steinshark/Pictures/training_data/flood/":("flood",333),
                                        "C:/users/Steinshark/Pictures/training_data/gobi/":("gobi",521),
                                        "C:/users/Steinshark/Pictures/training_data/shanghai/":("shanghai",524),
                                      },
                                      n_iters=n_iters,
                                      load_n=load_n)


    def train_epoch(self,bs=16):

        dataloader              = DataLoader(self.dataset,batch_size=bs,shuffle=True)
        self.losses[self.epoch] = []


        for i,batch in enumerate(dataloader):   

            #print(f"\t{i}/{len(dataloader)}")
            
            #Zero grad 
            for param in self.model.parameters():
                param.grad  = None 

            #Load data
            x               = batch[0].to(self.device).type(torch.float)
            y               = batch[1].type(torch.LongTensor)
            y               = y.type(torch.float).to(self.device)

            #Forward pass
            prediction      = self.model.forward(x)

            #Backward pass 
            loss            = self.loss_fn(prediction,y)
            loss.backward()
            self.losses[self.epoch].append(loss.mean())
            print(f"\tloss={sum(self.losses[self.epoch])/len(self.losses[self.epoch])}")

            #Step
            self.optimizer.step()
        

        self.epoch += 1


    def check_ch(self,ch_n,img,scale=1):

        #Put image through layer1
        img     = img.type(torch.float).to(t.device)
        with torch.no_grad():
            img     = self.model.conv1(img)[ch_n]

        if scale:
            img     = img - torch.min(img)
            img     = img / torch.max(img)
            img     = img * scale

        img     = torch.stack([img,img,img])
        #img     = self.to_pil_img(img)

        return img 


    def make_grid(self,img=torch.load("C:/users/steinshark/pictures/training_data/flood/176.tsr")[2]):
        imgs        = [] 
        base        = img.type(torch.float).to(t.device)

        for i in range(16):
            img     = self.check_ch(i,base)
            imgs.append(img)
        
        imgs    = torch.stack(imgs)
        grid    = torchvision.utils.make_grid(imgs,nrow=4).cpu().numpy()
        grid    = numpy.transpose(grid,(1,2,0))
        return grid


    def place_grid(self,imgs,nrow=4):
        imgs    = torch.stack(imgs)
        grid    = torchvision.utils.make_grid(imgs,nrow=nrow).cpu().numpy()
        grid    = numpy.transpose(grid,(1,2,0))
        return grid

if __name__ == "__main__":

    t   = Trainer() 
    t.select_model()

    imgs    = {}

    for _ in range(9):
        print(f"run epoch: {t.epoch}")
        t.select_dataset(n_iters=32,load_n=4)
        t.train_epoch(bs=4)
        imgs[t.epoch]   = t.check_ch(0,torch.load("C:/users/steinshark/pictures/training_data/flood/176.tsr")[2])
    

    #Get 16 ch
    in_img  = torch.load("C:/users/steinshark/pictures/training_data/flood/176.tsr")[2].type(torch.float).to(t.device)
    grid    = t.place_grid(list(imgs.values()),nrow=3)
    plt.imshow(grid)
    plt.title(f"Epochs: 0-9 -> class={t.model.forward(in_img)}")
    plt.show()


    