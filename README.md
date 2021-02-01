# Trackjectory
## Tracking and Trajectory Prediction 

Components Implemented
1. Siamese Tracker for online visual tracking.
2. FairMOT for multi-object detection and tracking. (<a href="https://medium.com/analytics-vidhya/fairmot-multi-object-tracking-386afe930b24">FairMOT: Multi-Object Tracking</a>)
3. Social GCN for trajectory forcasting.
<br>
[1] is implemented to identify and track the template image provided as input across the video frames. Its a single object tracking mechanism.<br>
[2] is a single shot MOT tracker used to predict multiple objects across frames while tracking them simultaneously. After extracting the tracked objects, those outputs are furnished to [3] to predict the trajectories.

The code is straightforward and easy to follow.

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

# Installation
1. Run following to clone into local system `https://github.com/Sai-Venky/Trackjectory.git`.
1. Create conda environment `conda create --name track python=3.6` and activate it `conda activate track`.
2. Run `pip install -r requirements.txt`.
3. Run `pip install lap cython-bbox`
3. Setup DCNv2 by going into `cd DCNV2` and running `sh ./make.sh`.


### Dataset

The dataset can be downloaded from the following links :-

`http://sintel.is.tue.mpg.de/`

The dataset can be extracted and stored in the parent directory. If not, its location can be changed in `src/utils/config.py` at `dataset_dir`

### Directory Layout

The directory structure is as follows :-

* **data :** contains the necessary files needed for loading the MPI Sintel dataset along with transformation functions.
  * dataset : base class which instantiates the mpi sintel dataset and transforms the raw data.
  * util : helper functions for preprocessing the image, flows
  * mpi_sintel : contains class needed to process/parse the mpi dataset data
* **models :** this contains the optical flow models and all of its constituent methods.
    * optical_flow : core model with train, validate functions. Instantiates and Calls all other models (flownet, loss).
    * flow_net2SD : contains the flow net SD model
    * loss : contains the loss functions used by model
* **utils :** this contains the utility methods needed.
    * config : contains the configuration/options.

 ### Results
#### Multi Object Tracking
![Multi Object Tracking](out/track.gif)

#### Trajectory Forcasting Output
![Trajectory Forcasting Output ](out/traj.gif)

#### Single Object Tracking Output and Template
<img src="out/sot.png" width="635" height="350" alt="Single Object Tracking Output ">&nbsp;&nbsp;<img src="out/sot_template.png" width="110" height="170" alt="Single Object Tracking Template">

 ### Acknowledgement

https://github.com/ifzhang/FairMOT
https://github.com/zllrunning/SiameseX.PyTorch
https://github.com/abduallahmohamed/Social-STGCNN
 ### Contributing

 You can contribute in serveral ways such as creating new features, improving documentation etc.

 ### Licence

 MIT Licence
