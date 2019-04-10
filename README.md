# AI-Pianist
A computer vision project that aims to generate vivid and accurate piano sound based on muted piano playing videos.


## Dataset

The current dataset is an external dataset consisting of muted video and midi audio. We firstly converted the `.mid` audio to `.wav` analog audio, and the create a `.txt` of audio offset against the video start time.

The 2-line txt format is 

    <framerate (fps)>
    <offset (hour:minute:second:#frame)>

The `.wav` and `.txt` files are named by appending the extension suffix to `x.wmv`, i.e. `1.wmv.txt`. They are located next to the corresponding `.wmv` and the `.wmv.mid` files.
