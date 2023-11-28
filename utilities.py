import torch 
import random 
import torchvision 

MULT    = 255/2

def generate_dataset(data_dict,window=3,dark_thresh=-.95):

    for classification in data_dict:
            print(f"start {classification}")
            saved           =  0
            path_load       = data_dict[classification][0]
            path_save       = data_dict[classification][1]

            for i in range(10_000_000):
                
                if i % 50 == 0:
                    print(f"\ti={i}\n")
                
                video       = torchvision.io.read_video(path_load,pts_unit='sec',start_pts=i*window ,end_pts=i*window + window,output_format="TCHW")[0].type(torch.float32)
                video.requires_grad_(False)

                if video.shape[0] == 1:
                    break 
                else:

                    #Cut to 512x512
                    dims        = 512
                    h           = video.shape[2]
                    w           = video.shape[3]
                    w_start     = int(int(w/2) - int(dims/2))
                    h_start     = int(int(h/2) - int(dims/2))
                    video       = video[:,:,h_start:h_start+dims,w_start:w_start+dims]

                    #Normalize 
                    video       = video / MULT
                    video       = video - 1 

                    #Save
                    for index in range(video.shape[0]):

                        cur_data    = video[index].type(torch.float16)

                        #If video mean < thresh dont even bother 
                        if torch.mean(cur_data) < dark_thresh:
                            continue
                        else:
                            #print(f"save data shape: {cur_data.shape}")
                            torch.save(cur_data,path_save+f"/{classification}{saved}.tsr")
                            saved += 1

def ch_visualize(img):
     pass


if __name__ == "__main__":
     dictionary     = {  
                                        #"flood":["C:/users/Steinshark/Pictures/training_data/flood_clip.mp4","C:/data/battlefield/flood"],
                                        #"gobi":["C:/users/Steinshark/Pictures/training_data/gobi_clip.mp4","C:/data/battlefield/gobi"],
                                        #"shanghai":["C:/users/Steinshark/Pictures/training_data/shanghai_clip.mp4","C:/data/battlefield/shanghai"]
                                        # "zavod":["C:/users/Steinshark/Pictures/training_data/zavod_clip.mp4","C:/data/battlefield/zavod"]
                                        # "islands":["C:/users/Steinshark/Pictures/training_data/islands_clip.mp4","C:/data/battlefield/islands"]
                                        "dawn":["C:/users/Steinshark/Pictures/training_data/dawn_clip.mp4","C:/data/battlefield/dawn"]

                                      }
     generate_dataset(dictionary)