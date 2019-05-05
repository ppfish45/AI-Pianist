from mido import MidiFile
import mido

mid = MidiFile('1.wmv.mid')

new_mid_1 = MidiFile(ticks_per_beat=mid.ticks_per_beat)
# new_mid_2 = MidiFile(ticks_per_beat = mid.ticks_per_beat)
track1 = mido.MidiTrack()
# track2 = mido.MidiTrack()

new_mid_1.tracks.append(track1)
# new_mid_2.tracks.append(track2)
# start_time = 44.7
# stop_time = 55.3

time = 0.0
for tracks in mid.tracks:
    for msg in tracks:
        time += mido.tick2second(msg.time, mid.ticks_per_beat, 500000.0)
        # if (time < start_time):
        track1.append(msg)
        # elif(time > stop_time):
        # track2.append(msg)

new_mid_1.save('1.1.wmv.mid')
# new_mid_2.save('1.2.wmv.mid')
