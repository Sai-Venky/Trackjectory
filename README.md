# TrackJectory
## Tracking and Trajectory Prediction 

Components Implemented :-
1. Siamese Tracker for single object online visual tracking. This identify's and track's the template image provided as input across the video frames.<br>
2. FairMOT for one shot multi-object detection and tracking. (<a href="https://medium.com/analytics-vidhya/fairmot-multi-object-tracking-386afe930b24">FairMOT: Multi-Object Tracking</a>)
3. Social GCN for trajectory forecasting. FairMOT is used to predict and track multiple objects across frames. After extracting the tracked objects, those outputs are furnished to Social GCN to forecast the trajectories.


<ul>
<li><a href="https://github.com/Sai-Venky/Trackjectory#installation-and-running">Installation and Running</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#dataset">Dataset</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#directory-layout">Directory Layout</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#results">Results</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#acknowledgement">Acknowledgement</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#contributing">Contributing</a></li>
<li><a href="https://github.com/Sai-Venky/Trackjectory#licence">Licence</a></li>
</ul>

### Installation and Running

1. Run following to clone into local system `https://github.com/Sai-Venky/Trackjectory.git`.
1. Create conda environment `conda create --name track python=3.6` and activate it `conda activate track`.
2. Run `pip install -r requirements.txt`.
3. Run `pip install lap cython-bbox`
3. Setup DCNv2 by going into `cd DCNV2` and running `sh ./make.sh`.

#### Training and Testing

All configurable values with details on their significance are in `utils/config.py`.

**Single Object Tracking** :- <br>
  Training - `python src/siam_train.py`<br>
  Testing - `python src/siam_track.py`<br><br>
**Multi Object Tracking** :-<br>
  Training - `python src/mot_train.py`<br>
  Testing - `python src/mot_track.py`<br><br>
**Trajectory Forcasting** :-<br>
  Training - `python src/trajectory_train.py`<br>
  Testing - `python src/trajectory_test.py`<br><br>
  
### Dataset

The dataset is created from Kabaddi player videos curated from multiple online platforms. (https://youtu.be/HOfY9g05Sv4)
This was selected since this sport depicts a lot of movements (feints) and is challenging for forcasting trajectory correctly.

In order to train with custom dataset, change the config value of `images_dataset`, `single_track`, `trajectory_dataset` in `utils/config.py`

### Directory Layout

The directory structure is as follows :-

* **dataset :** contains the necessary files needed for loading the Siamese, MOT and Trajectory forcasting datasets along with transformation functions.
  * mot_dataset : training/validation and testing base class which instantiates the fairmot dataset and transforms the raw data with affine transformation.
  * siam_dataset : training/validation base class which instantiates the siamese dataset and returns the template with target frame.
  * trajectory_dataset : training/validation base class which instantiates the trajectory dataset after constructing the graph based on tracked objects and frame sequences.
  * util : helper functions for preprocessing the image.
* **models :** this contains the base models for TrackJectory with all of its constituent methods.
    * gcn : contains the social graph cnn model with time extrapolator cnn.
    * pose_dla_dcn : contains the dla-34 model with deformable convolutions initialization.
    * single_track : contains the siam rpn++ model (resnet) with depthwise correlation.
    * loss : contains the mot, bivariate and L1 loss functions.
* **utils :** this contains the utility methods for TrackJectory.
    * config : contains the configuration options.
    * scheduler : contains the scheduler options for adjusting learning rate dynamically.
    * utils : contains the load, save model functions for TrackJectory.
    * decode : contains the util functions for extracting the feasible outputs during multiple object tracking.
* **tracker :** this contains the tracker instantiations for performing live inference.
    * basetrack : the base class for multi object tracking.
    * matching : contains the methods needed for fusing the iou, embedding and Kalman Filter (KF) outputs.
    * multitracker : implementation of Joint Detection and Embedding (JDE) tracker for performing multi object tracking and association.
    * singletracker : implementation of a SiamRPN++ tracker for performing single object tracking.
* **tracking_utils :** this contains the utils needed for the multi object tracker.
    * kalman_filter : contains the methods for predicting, updating KF values and computing gating distance between KF outputs and embedding distance.
    * log : logger functions.
    * timer : timer functions.
    * visualization : methods for visualizing the multiple outputs predicted by JDE tracker.
* mot_track : main file performing tasks for tracking multiple objects and creating the inputs for trajectory forcasting.
* mot_train : main file performing tasks for training FairMOT.
* siam_track : main file performing tasks for tracking a single object.
* siam_train : main file performing tasks for training SiamRPN++.
* trajectory_test : main file performing tasks for testing Social GCN and visualization functions for forming the trajectories based on outputs.
* trajectory_train : main file performing tasks for training Social GCN with outputs predicted from multi object tracker.

 ### Results
#### Multi Object Tracking
![Multi Object Tracking](out/track.gif)

#### Trajectory Forcasting Output
![Trajectory Forcasting Output ](out/traj.gif)

#### Single Object Tracking Output and Template
<img src="out/sot.png" width="635" height="350" alt="Single Object Tracking Output ">&nbsp;&nbsp;<img src="out/sot_template.png" width="110" height="170" alt="Single Object Tracking Template">

 ### Acknowledgement

https://github.com/ifzhang/FairMOT<br>
https://github.com/zllrunning/SiameseX.PyTorch<br>
https://github.com/abduallahmohamed/Social-STGCNN<br>

 ### Contributing

 You can contribute in serveral ways such as creating new features, improving documentation etc.

 ### Licence

 MIT Licence
