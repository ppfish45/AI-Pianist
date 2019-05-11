# AI-Pianist
A computer vision project that aims to generate vivid and accurate piano sound based on muted piano playing videos.


## Dataset

### ~~Dataset for Pressed Key Detection~~

The current dataset is an external dataset consisting of muted video and midi audio. We firstly converted the `.mid` audio to `.wav` analog audio, and the create a `.txt` of audio offset against the video start time.

The 2-line txt format is 

    <framerate (fps)>
    <offset (hour:minute:second:#frame)>

The `.wav` and `.txt` files are named by appending the extension suffix to `x.wmv`, i.e. `1.wmv.txt`. They are located next to the corresponding `.wmv` and the `.wmv.mid` files.~~

### Dataset for Keyboard Localization

This dataset consists of multiple images containing __complete__ keyboards in it. For each image, we manually labelled the coordinates of four corners of the keyboard in the order of left-top, right-top, right-bottom and left-bottom.

To explain the format of this dataset, we take `X_train` and `y_train` as an example. There are several subdirectories under `X_train` named `0`, `1`, `2` and so on. In each subdirectory, multiple `jpg` images are stored.

Respectively, in `y_train`, there are the same number of ground truth files named `0.npy`, `1.npy`, `2.npy` and so on. Each `npy` file stores an array in the shape of `(N, 4, 2)`, which records the corner coordinates of each image under the corresponding subdirectory in `X_train`.

### Dataset for Note Correspondence

This dataset describes a corresponding relationship between all the frames and the note generated at those frames. 

Similar as the above dataset, we make two directories, `X_train` and `y_train`, to store the note labelling result. There are several subdirectories under `X_train` named  `1`, `2` and so on. In each subdirectory, multiple `jpg` images are stored. 

Correspondingly, in `y_train`, there are the same number of ground truth files named `0.npy`, `1.npy`, `2.npy` and etc to keep the note labels. Each `npy` file stores an array in the shape of `(N, 128)`, where `N` represents the number of frames in video `1`, and the second dimension stores the note information of certain frame. Since we have overall `128` note level, we take `128` to be the size of second dimension.

The dataset has been uploaded to [Google Drive](https://drive.google.com/file/d/1W8_82fD_WX3FIU61hWBkEEzb_1oR6Avc/view?usp=sharing).

### Visualization of the training process of `keypress_recognition`

There is one line `from torch.utils.tensorboard import SummaryWriter`, which is supported by **PyTorch 1.1.0+** only. If you have this newest version on your machine, remember to also run 

    pip install tb-nightly

Then the module can be successfully loaded. To see the results, run (assume you are in the root folder)

    cd models
    tensorboard --logdir=runs

or you can change the `--logdir` parameter according to your cwd. Note that *when the tensorboard host is running while you produce some new running logs, the new data is not reflected immediately. You have to restart the tensorboard.*

If you do not have PyTorch 1.1.0, you can

    pip install tensorboardX

and **also install tensorflow**. You need to replace the import statement with

    from tensorboardX import SummaryWriter

accordingly.