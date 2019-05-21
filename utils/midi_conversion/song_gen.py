#!/usr/local/bin/python3
import mido
import numpy as np
import sys

# Default speed information
tempo = 500000.0
ticks_per_beat = 500  # Default note length
frame_rate = 30


# Given a npy file, output a midi track
def reproduce(message, track):
    # N = number of total frames, D = number of total notes (128 respectively)
    N, D = message.shape
    prev_frame = np.zeros(shape=(D,))
    prev_time = 0
    for t in range(N):
        curr_frame = message[t].reshape(-1, )
        difference = curr_frame - prev_frame
        if np.count_nonzero(difference) != 0:
            curr_time = t
            for note in range(D):
                if difference[note]:
                    time = float(curr_time - prev_time) / float(frame_rate)
                    msg = mido.Message('note_off' if difference[note] == -1 else 'note_on', note=note,
                                       time=int(mido.second2tick(time, ticks_per_beat=ticks_per_beat, tempo=tempo)),
                                       velocity=76)
                    track.append(msg)
                    prev_time = curr_time
        prev_frame = curr_frame
    return track


# Smoothen the given track, to be confirmed
def smooth(track):
    return track


message = np.load(sys.argv[0])
song = mido.MidiFile(ticks_per_beat=ticks_per_beat)
track = mido.MidiTrack()
song.tracks.append(track)
track = reproduce(message=message, track=track)
track = smooth(track=track)
song.save('gen.mid')
