#!/usr/local/bin/python3
import os
import re
import sys
import mido
import numpy as np

rate = 24

def frame_extraction(name):
    os.system("nohup ffmpeg -i ./" + name + " -vf fps=24 ./test_1/%d.jpg > output.txt ")


def note_labelling(name):
    # f = open(name, "r")

    total_frame = 0
    fi = open("output.txt")
    for line in fi.readlines():
        if re.search("frame", line) is not None:
            total_frame = int(line[6:11])

    mid = mido.MidiFile(re.sub("MP4", "mid", name))
    label = np.zeros((total_frame, 128))
    prev_frame = int(0)
    curr_frame = 0
    timer = 0
    

    for tracks in mid.tracks:
        if (curr_frame > total_frame):
            break
        for msg in tracks:
            timer += msg.time
            curr_time = mido.tick2second(timer, mid.ticks_per_beat, 500000.0)
            curr_frame = curr_time * rate
            curr_frame = round(curr_frame)
            prev_frame = round(prev_frame)
            if (msg.type == "note_on"): 
                label[prev_frame:(curr_frame+1), :] = label[prev_frame, :] 
                label[curr_frame, msg.note] = msg.velocity
            elif (msg.type == "note_off"):
                label[prev_frame:(curr_frame+1), :] = label[prev_frame, :] 
                label[curr_frame, msg.note] = - msg.velocity
            prev_frame = curr_frame

        

    np.save("./test_2.npy", label)

    fi.close()

frame_extraction(sys.argv[1])
note_labelling(sys.argv[1])