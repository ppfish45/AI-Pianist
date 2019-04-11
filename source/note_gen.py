#!/usr/local/bin/python3

import re
import os
import sys
import mido
import numpy as np

def frame_extraction(root, txt_name, index):
    wmv_name = re.sub(".txt", "", txt_name)
    os.mkdir("../DataSet/X_train/"+str(index))
    os.system("nohup ffmpeg -i "+ os.path.join(root, wmv_name) + " -vf fps=30 ../DataSet/X_train/"+str(index)+"/%d.jpg > output.txt ")


def note_labelling (root, txt_name,index):
    f = open (os.path.join(root,txt_name), "r")

    total_frame = 0
    fi = open("output.txt")
    for line in fi.readlines():
        if (re.search("frame", line)!=None):
            total_frame = int(line[6:11])

    rate = int(f.readline()) # frame rate, # of frames per second
    hour, minute, second, frame = f.readline().split(':')
    frame_time =  3600 * int (hour) * rate + 60 * int (minute) * rate + int (second) * rate + int(frame)

    mid_name = re.sub("txt", "mid", txt_name)
    mid = mido.MidiFile(os.path.join(root, mid_name))

    label = np.zeros((total_frame,128))
    boolean = np.zeros(128)

    for tracks in mid.tracks:
        for msg in tracks:
            if (msg.type=="note_on"):
                boolean[int(msg.note)]=1
            elif (msg.type=="note_off"):
                boolean[int(msg.note)]=0
            msg_time = mido.tick2second(msg.time,mid.ticks_per_beat,500000.0)
            for i in range(128):
                if (boolean[i]==1):
                    label[frame_time:frame_time+int(msg_time*rate), i] = 1
                elif (boolean[i]==0):
                    label[frame_time:frame_time+int(msg_time*rate), i] = 0
            frame_time += int(msg_time*rate)
            
    np.save("../DataSet/y_train/"+str(index)+'.npy', label)
        
    # close file
    fi.close()
    f.close()

# open the wmv info file
i = 0
for root, dirs, files in os.walk("../DataSet/TrainSet"):
    for file in files:
        if (re.search(".txt", file)!=None):
            i+=1
            frame_extraction(root, file, i)
            note_labelling(root, file, i)

# # test case
# root = "../DataSet/TrainSet/1/"
# file = "1.wmv.txt"
# if (re.search(".txt", file)!=None):
#     i+=1
#     frame_extraction(root, file, i)
#     note_labelling(root, file, i)