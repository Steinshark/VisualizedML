import torch 
import models 
from torch.utils.data import Dataset, DataLoader
import torchvision 
from matplotlib import pyplot as plt
import random 
import time 
import torchvision.utils
import numpy 
import os 
import math



FLOODIMG1    = (torch.load("C:/data/battlefield/flood/flood3045.tsr") + 1) / 2 
FLOODIMG2    = (torch.load("C:/data/battlefield/flood/flood7933.tsr") + 1) / 2 
FLOODIMG3    = (torch.load("C:/data/battlefield/flood/flood2949.tsr") + 1) / 2 
FLOODIMG4    = (torch.load("C:/data/battlefield/flood/flood5109.tsr") + 1) / 2 
GOBIIMG1    = (torch.load("C:/data/battlefield/gobi/gobi3312.tsr") + 1) / 2 
GOBIIMG2    = (torch.load("C:/data/battlefield/gobi/gobi4412.tsr") + 1) / 2 
GOBIIMG3    = (torch.load("C:/data/battlefield/gobi/gobi2587.tsr") + 1) / 2 
GOBIIMG4    = (torch.load("C:/data/battlefield/gobi/gobi3044.tsr") + 1) / 2 
SHANGIMG1    = (torch.load("C:/data/battlefield/shanghai/shanghai3458.tsr") + 1) / 2 
SHANGIMG2    = (torch.load("C:/data/battlefield/shanghai/shanghai1376.tsr") + 1) / 2 
SHANGIMG3    = (torch.load("C:/data/battlefield/shanghai/shanghai430.tsr") + 1) / 2 
SHANGIMG4    = (torch.load("C:/data/battlefield/shanghai/shanghai1532.tsr") + 1) / 2 
ZAVODIMG1    = (torch.load("C:/data/battlefield/zavod/zavod12727.tsr") + 1) / 2 
ZAVODIMG2    = (torch.load("C:/data/battlefield/zavod/zavod2614.tsr") + 1) / 2 
ZAVODIMG3    = (torch.load("C:/data/battlefield/zavod/zavod2465.tsr") + 1) / 2 
ZAVODIMG4    = (torch.load("C:/data/battlefield/zavod/zavod14147.tsr") + 1) / 2 
ISLANDSIMG1    = (torch.load("C:/data/battlefield/islands/islands4551.tsr") + 1) / 2 
ISLANDSIMG2    = (torch.load("C:/data/battlefield/islands/islands7698.tsr") + 1) / 2 
ISLANDSIMG3    = (torch.load("C:/data/battlefield/islands/islands1986.tsr") + 1) / 2 
ISLANDSIMG4    = (torch.load("C:/data/battlefield/islands/islands9853.tsr") + 1) / 2 
DAWNIMG1    = (torch.load("C:/data/battlefield/dawn/dawn6463.tsr") + 1) / 2 
DAWNIMG2    = (torch.load("C:/data/battlefield/dawn/dawn18581.tsr") + 1) / 2 
DAWNIMG3    = (torch.load("C:/data/battlefield/dawn/dawn13030.tsr") + 1) / 2 
DAWNIMG4    = (torch.load("C:/data/battlefield/dawn/dawn19292.tsr") + 1) / 2 


img_list = {"flood1":FLOODIMG1,
            "flood2":FLOODIMG2,
            "flood3":FLOODIMG3,
            "flood4":FLOODIMG4,
            "gobi1":GOBIIMG1,
            "gobi2":GOBIIMG2,
            "gobi3":GOBIIMG3,
            "gobi4":GOBIIMG4,
            "shanghai1":SHANGIMG1,
            "shanghai2":SHANGIMG2,
            "shanghai3":SHANGIMG3,
            "shanghai4":SHANGIMG4,
            "zavod1":ZAVODIMG1,
            "zavod2":ZAVODIMG2,
            "zavod3":ZAVODIMG3,
            "zavod4":ZAVODIMG4,
            "islands1":ISLANDSIMG1,
            "islands2":ISLANDSIMG2,
            "islands3":ISLANDSIMG3,
            "islands4":ISLANDSIMG4,
            "dawn1":DAWNIMG1,
            "dawn2":DAWNIMG2,
            "dawn3":DAWNIMG3,
            "dawn4":DAWNIMG4}

#Good is 3045,7933,2949,5109
class imgDataSet(Dataset):
    
    def __init__(self,path,load_n=8,save_mem=False):

        self.data           = [] 

        #Find classes
        self.classes        = os.listdir(path)
        self.paths          = [(path+"/"+indv_class+"/"+indv_class).replace("//","/") for indv_class in self.classes]
        
        for _ in range(load_n):
            
            #Load file 
            base_path       = random.choice(self.paths)
            classification  = base_path.split("/")[-1]
            path            = "/".join(base_path.split("/")[:-1])
            filename        = path + "/" + random.choice(os.listdir(path))
            tensor          = torch.load(filename).type(torch.float16)

            #Create class tensor 
            class_t         = torch.zeros(len(self.paths),dtype=torch.float)
            i               = self.classes.index(classification) 
            class_t[i]      = 1

            self.data.append([tensor,class_t])




            
                
                
                
                

            
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
        self.percents   = []
        self.to_pil_img = torchvision.transforms.ToPILImage()


    def select_model(self,lr=.0001,n_classes=5):
        self.model      = models.visNet(n_classes=n_classes).to(self.device)
        self.optimizer  = torch.optim.Adam(self.model.parameters(),lr=lr,weight_decay=lr/10,betas=(.75,.9999))


    def select_dataset(self,load_n=4):
        self.dataset    = imgDataSet("C:/data/battlefield/",load_n=load_n)
        self.n_classes  = len(self.dataset.classes)
        self.accuracies = torch.zeros(size=(self.n_classes,self.n_classes))


    def train_epoch(self,bs=16):

        dataloader              = DataLoader(self.dataset,batch_size=bs,shuffle=True)
        self.losses[self.epoch] = []

        #For printing 
        num_equals 	    = 50 
        printed 	    = 0
        num_batches     = len(dataloader)
        print(f"\t\t[",end='')


        for i,batch in enumerate(dataloader):   
            
            batch_acc   = torch.zeros(size=(self.n_classes,self.n_classes))
            #print(f"\t{i}/{len(dataloader)}")
            percent     = i / num_batches

            while (printed / num_equals) < percent:
                    print("=",end='',flush=True)
                    printed+=1
            
            #Zero grad 
            self.model.zero_grad() 

            #Load data
            x               = batch[0].to(self.device).type(torch.float32)
            y               = batch[1].to(self.device).type(torch.float32)

            real_indices    = torch.max(y,dim=1)[1]

            #Forward pass
            prediction      = self.model.forward(x)

            pred_indices    = torch.max(prediction,dim=1)[1]

            #print(f"y=\n{y}\n\npred=\n{prediction}\n\nmaxs=\n{pred_indices}")
            for real_i,pred_i in zip(real_indices,pred_indices):
                self.accuracies[real_i][pred_i] += 1
                batch_acc[real_i][pred_i] += 1
                #input(f"accuracies=\n{self.accuracies}")
            #Backward pass 
            loss            = self.loss_fn(prediction,y)
            loss.backward()
            self.losses[self.epoch].append(loss.mean().detach().item())
            batch_accuracy = (100*torch.sum(torch.diag(torch.ones(self.n_classes)) * batch_acc)/torch.sum(batch_acc)).detach().item() / 100
            self.percents.append(batch_accuracy)

            #Step
            self.optimizer.step()

        
        correct                     = torch.diag(torch.ones(self.n_classes)) * self.accuracies
        #self.percents.append(correct) 
        print(f"]\tloss={sum(self.losses[self.epoch])/len(self.losses[self.epoch])}")
        print(f"\t\taccuracy={100*torch.sum(correct)/torch.sum(self.accuracies):.2f}%")

        

        self.epoch += 1


    def layer_to_grid(self,img=FLOODIMG2,layer=1,nrow=6):
        base     = img.type(torch.float).to(t.device)
        imgs     = self.model.view_layer(base,layer)
        
        imgs    = torch.stack(imgs)
        grid    = torchvision.utils.make_grid(imgs,nrow=nrow).cpu().numpy()
        grid    = numpy.transpose(grid,(1,2,0))
        return grid


    def place_grid(self,imgs,nrow=6,bypass_numpy=False):
        imgs    = torch.stack(imgs)
        grid    = torchvision.utils.make_grid(imgs,nrow=nrow).cpu().numpy()

        if bypass_numpy:
            return torch.from_numpy(grid)

        grid    = numpy.transpose(grid,(1,2,0))
        return grid


    def store_model(self,store_path="model_ckpts",late=True):
        
        #Create if non-existent 
        if not os.path.isdir(store_path):
            os.mkdir(store_path)
        
        #Save model there 
        torch.save(self.model.state_dict(),f"{store_path}/model_ckpt_{self.epoch - 1 if late else 0}.mdl")


    def load_model(self,epoch,store_path="model_ckpts"):
        self.model.load_state_dict(torch.load(f"{store_path}/model_ckpt_{epoch}.mdl"))


    def get_n_epochs(self,store_path="model_ckpts"):
        return len(os.listdir(store_path))


if __name__ == "__main__":

    t   = Trainer() 
    # while True:
    #     rand_i  = random.randint(19000,20020)
    #     img     = (torch.load(f"C:/data/battlefield/dawn/dawn{rand_i}.tsr") + 1) / 2 
    #     img     = t.to_pil_img(img)
    #     print(f"i={rand_i}")
    #     plt.imshow(img)
    #     plt.show()


    t.select_model(lr=.00005,n_classes=6)
    n_layers    = 8
    dataset     = {l:[] for l in range(n_layers)}
    loading     = 1024
    iters       = 1
    bs          = 32
    # t.select_dataset(load_n=loading)
    # for _ in range(iters):
    #     print(f"\trun epoch: {t.epoch}")
    #     t.train_epoch(bs=bs)
    #     imgs[0].append(t.model.view_layer(FLOODIMG2,layer=1))
    #     imgs[1].append(t.model.view_layer(FLOODIMG2,layer=2))

    


    while True:
        command     = input(f"\nin <: ")

        commands    = command.split(" ")
        if '' in commands:
            commands.remove('')

        if "epoch" in commands[0]:
            
            #Ensure proper command
            if not len(commands) > 2:
                print("format: epoch <epoch> <layer>")
            else:
                ep          = int(commands[1])
                layer       = int(commands[2])
                img         = img_list[commands[3] if len(commands) > 3 and commands[3] in img_list else 'flood2']

                if layer > len(t.model.layers):
                    print(f"layer '{layer}' is out of bounds for [0,{len(t.model.layers)}]")
                if ep > t.epoch-1:
                    print(f"epoch '{ep}' is out of bounds for [0,{t.epoch-1}]")
                else:

                    #Load corresponding epoch
                    t.load_model(ep)
                    img_set     = t.model.view_layer(img,layer)

                    #Grid and view
                    grid        = t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))))
                    plt.imshow(grid)
                    plt.title(f"Ch. All, Ep. {ep}, Layer {layer}")
                    plt.show()

        elif "ch" in commands[0]:
            if not len(commands) > 2:
                print("format: ch <ch> <layer>")
            else:
                ch          = int(commands[1])
                layer       = int(commands[2])
                img         = img_list[commands[3] if len(commands) > 3 and commands[3] in img_list else 'flood2']

                if layer > len(t.model.layers):
                    print(f"layer '{layer}' is out of bounds for [0,{len(t.model.layers)}]")

                #Compile imgs 
                img_set     = [] 
                for ep_i in range(t.epoch):
                    t.load_model(ep_i)
                    img_set.append(t.model.view_layer(img,layer)[ch])

                #Grid and view
                grid    = t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))))
                plt.imshow(grid)
                plt.title(f"Ch. {ch}, Ep. All, Layer {layer}")
                plt.show()
        
        elif "exit" in commands[0] or "quit" in commands[0]:
            exit()
        
        elif "train" in commands[0]:

            if len(commands) > 1:
                iters     = int(commands[1])

            for _ in range(iters):
                t.select_dataset(load_n=loading)
                print(f"\ttrain epoch: {t.epoch}")
                t.train_epoch(bs=bs)
                for layer in range(n_layers):
                    epoch_layer = t.model.view_layer(FLOODIMG2,layer=layer+1)
                    dataset[layer].append(epoch_layer)
                
                t.store_model()

        elif "test" in commands[0]:
            imgs    = [FLOODIMG1,FLOODIMG2,FLOODIMG3,FLOODIMG4,GOBIIMG1,GOBIIMG2,GOBIIMG3,GOBIIMG4,SHANGIMG1,SHANGIMG2,SHANGIMG3,SHANGIMG4,ZAVODIMG1,ZAVODIMG2,ZAVODIMG3,ZAVODIMG4]
            imgs    = [img.type(torch.float) for img in imgs] 
            random.shuffle(imgs)

            #Send 4 at a time
            for i in range(4):
                start           = 4*i 
                in_tsr          = torch.stack(imgs[start:start+4]).type(torch.float).to(t.device)
                predictions     = t.model.forward(in_tsr)

                torch.set_printoptions(sci_mode=False)
                pred0           = [f"{k[:2]}:{100*p.item():.1f}%" for k,p in zip(t.dataset.classes,predictions[0])]
                pred1           = [f"{k[:2]}:{100*p.item():.1f}%" for k,p in zip(t.dataset.classes,predictions[1])]
                pred2           = [f"{k[:2]}:{100*p.item():.1f}%" for k,p in zip(t.dataset.classes,predictions[2])]
                pred3           = [f"{k[:2]}:{100*p.item():.1f}%" for k,p in zip(t.dataset.classes,predictions[3])]

                title           = f"Classes: {t.dataset.classes}\n{pred0}     {pred1}     {pred2}     {pred3}"

                grid            = t.place_grid(imgs[start:start+4],nrow=4)
                plt.imshow(grid)
                plt.title(title)
                plt.show()

        elif "set" in commands[0]:

            variable    = commands[1]
            val         = commands[2]

            if "iters" in variable:
                iters     = int(val)
            elif "loading" in variable:
                loading     = int(val)
            elif 'bs'   in variable:
                bs          = int(val)
            else:
                print(f"\tno var found: '{variable}")
            
        elif "var" in commands[0]:
            print(f"\tloading\t{loading}\n\titers\t{iters}\n\tbs\t{bs}")

        elif "plot" in commands[0]:

            if not len(commands) > 1:
                print(f"format: plot <variable>")
            else:
                if "loss" in commands[1]:
                    data    = sum(list(t.losses.values()),[])
                    plt.plot(data,color='darkorange',label="loss")
                    plt.legend()
                    plt.title(f"Loss vs epoch")
                    plt.show()
                elif "accur" in commands[1]:
                    data    = t.percents
                    plt.plot(data,color='dodgerblue',label="accuracy")
                    plt.legend()
                    plt.title(f"Accuracy vs epoch")
                    plt.show() 
                elif 'both' in commands[1]:
                    loss    = sum(list(t.losses.values()),[])
                    accu    = t.percents
                    plt.plot(loss,color='darkorange',label="loss")
                    plt.plot(accu,color='dodgerblue',label="accuracy")
                    plt.legend()
                    plt.title("Accuracy and Loss vs epoch")
                    plt.show()

        elif 'save' in commands[0]:
            if len(commands) < 2:
                commands.append("1")
            
            #Save base image 
            img         = FLOODIMG2
            base_img    = t.to_pil_img(img)
            base_img.save("BaseImg.jpg")


            #Save layer 1 
            layer1_img  = t.to_pil_img(t.place_grid(t.model.view_layer(img,1),nrow=4,bypass_numpy=True))
            layer1_img.save("Layer1_ep0.jpg")

            #Save last epoch 
            img_set     = [] 
            for ep_i in range(t.epoch):
                t.load_model(ep_i)
                img_set.append(t.model.view_layer(img,1)[int(commands[1])])
            layer1_img  = t.to_pil_img(t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))),bypass_numpy=True))
            layer1_img.save("Layer1_ch1.jpg")

            #Save last epoch 
            img_set     = [] 
            for ep_i in range(t.epoch):
                t.load_model(ep_i)
                img_set.append(t.model.view_layer(img,2)[int(commands[1])])
            layer1_img  = t.to_pil_img(t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))),bypass_numpy=True))
            layer1_img.save("Layer2_ch1.jpg")

            #Save last epoch 
            img_set     = [] 
            for ep_i in range(t.epoch):
                t.load_model(ep_i)
                img_set.append(t.model.view_layer(img,3)[int(commands[1])])
            layer1_img  = t.to_pil_img(t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))),bypass_numpy=True))
            layer1_img.save("Layer3_ch1.jpg")

            #Save last epoch 
            img_set     = [] 
            for ep_i in range(t.epoch):
                t.load_model(ep_i)
                img_set.append(t.model.view_layer(img,8)[int(commands[1])])
            layer1_img  = t.to_pil_img(t.place_grid(img_set,nrow=math.ceil(math.sqrt(len(img_set))),bypass_numpy=True))
            layer1_img.save("Layer8_ch1.jpg")
