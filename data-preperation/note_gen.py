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
    # timer = np.zeros(128)
    # delta_frame = int(mido.tick2second(501,mid.ticks_per_beat,500000.0) * rate)

    for tracks in mid.tracks:
        for msg in tracks:
            # # update timer
            # timer[label[frame_time]==1] += msg.time
            # timer[label[frame_time]==0] = 0
            # get msg time in sec
            msg_sec_time = mido.tick2second(msg.time,mid.ticks_per_beat,500000.0)
            # broadcast numpy array
            # mask = label == 1
            label[frame_time:frame_time+int(msg_sec_time*rate), :] = label[frame_time, :]
            # if int(msg.time)> 500 :
            #     label[frame_time+delta_frame:frame_time+int(msg_sec_time*rate) , mask[frame_time] == 1] = 0
            # update frame time
            frame_time += int(msg_sec_time*rate)
            if (frame_time >= total_frame):
                frame_time = total_frame - 1
            # # first iter < 500 < second iter
            # for j in range(128):
            #     if (timer[j]>500):
            #         delta = int(mido.tick2second(timer[j],mid.ticks_per_beat,500000.0) * rate)
            #         label[frame_time-delta:frame_time, j] = 0
            # mark the coming frame
            if (msg.type=="note_on"):
                label[frame_time, msg.note] = 1
            elif (msg.type=="note_off"):
                label[frame_time, msg.note] = 0

            
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