import mido

song_1 = mido.MidiFile('video1.mid')
song_2 = mido.MidiFile('gen.mid')

for tracks in song_1.tracks:
    for msg in tracks:
        print(msg)

print ('------------------------------')

for tracks in song_2.tracks:
    for msg in tracks:
        print(msg)