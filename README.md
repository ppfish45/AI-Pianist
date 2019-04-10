# AI-Pianist
A computer vision project that aims to generate vivid and accurate piano sound based on muted piano playing videos.


## Dataset

### Dataset for Pressed Key Detection

The current dataset is an external dataset consisting of muted video and midi audio. We firstly converted the `.mid` audio to `.wav` analog audio, and the create a `.txt` of audio offset against the video start time.

The 2-line txt format is 

    <framerate (fps)>
    <offset (hour:minute:second:#frame)>

The `.wav` and `.txt` files are named by appending the extension suffix to `x.wmv`, i.e. `1.wmv.txt`. They are located next to the corresponding `.wmv` and the `.wmv.mid` files.

### Dataset for Keyboard Localization

This dataset consists of multiple images containing __complete__ keyboards in it. For each image, we manually labelled the coordinates of four corners of the keyboard in the order of left-top, right-top, right-bottom and left-bottom.

To explain the format of this dataset, we take `X_train` and `y_train` as an example. There are several subdirectories under `X_train` named `0`, `1`, `2` and so on. In each subdirectory, multiple `jpg` images are stored.

Respectively, in `y_train`, there are the same number of ground truth files named `0.npy`, `1.npy`, `2.npy` and so on. Each `npy` file stores an array in the shape of `(N, 4, 2)`, which records the corner coordinates of each image under the corresponding subdirectory in `X_train`.
