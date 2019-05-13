#!/usr/local/bin/python3
import mido
import numpy as np
import sys

# Default speed information
tempo = 500000.0
ticks_per_beat = 480  # Default note length
frame_rate = 24


# Given a npy file, output a midi track
def reproduce(message, track):
    # N = number of total frames, D = number of total notes (128 respectively)
    N, D= message.shape
    prev_frame = np.zeros(shape=(D,))
    prev_time = 0
    for t in range(N):
        curr_frame = message[t]
        difference = curr_frame - prev_frame
        if np.count_nonzero(difference) != 0:
            curr_time = t
            for note in range(D):
                if difference[note] != 0:
                    time = float(curr_time - prev_time) / float(frame_rate)
                    msg = mido.Message('note_off' if difference[note] < 0  else 'note_on', note=note,
                                        time=int(mido.second2tick(time, ticks_per_beat=ticks_per_beat, tempo=tempo)),
                                        velocity=70 if curr_frame[note]>0 else 0)
                    track.append(msg)
                    prev_time = curr_time
        prev_frame = curr_frame
    return track


# Smoothen the given track, to be confirmed
def smooth(track):
    return track


message_1 = np.load(sys.argv[1])
message_1 = message_1.astype('uint8')
message_2 = np.load(sys.argv[2])
message_2 = message_2.astype('uint8')
message = np.concatenate((message_1, message_2), axis=0)
song = mido.MidiFile(ticks_per_beat=ticks_per_beat)
track = mido.MidiTrack()
song.tracks.append(track)
track = reproduce(message=message, track=track)
track = smooth(track=track)
song.save('gen_2.mid')
