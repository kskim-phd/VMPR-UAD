# 3D unsupervised anomaly detection through virtual multi-view projection and reconstruction: Clinical validation of low-dose chest computed tomography(VMPR-UAD)

We release VMPR-UAD evaluation code.

Collaborators: Kyung-su Kim, Seongje Oh, Juhwan Lee 

Detailed instructions for testing the image are as follows.

------

# Implementation

A PyTorch implementation of VMPR-UAD based on original Patchcore and Automated lung segmentation code.

Patchcore[https://github.com/hcw-00/PatchCore_anomaly_detection] (Thanks for Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler.)

Automated lung segmentation in CT under presence of severe pathologies[https://github.com/JoHof/lungmask] (Thanks for Hofmanninger, J., Prayer, F., Pan, J. et al.)

------
## Environments

The setting of the virtual environment we used is described as requirement.txt.

------
## Segmentation

Put the test data in the "dataset" folder to create a split mask. The pre-trained split model weight file is automatically downloaded. Please run "Segmentation/inference.py".

```
python inference.py 
```
The segment mask (same name) is stored in the "dataset/segmentation" folder.

Download sample test data[[link](https://drive.google.com/file/d/1xQNQlHvg3HNWhgA_fpORc8L7-7h9jWst/view?usp=sharing)]

------
## Virtual multi-view projection

Please run "Multi_view_projection/project.py"

```
python project.py 
```
You will see that a "projection_data" folder is created and storing the multi-view projection images

------
## Anomaly detection

Download the embedings.zip from the link below, and extract it to Anomaly_detection/embeddings/. 

Please run "Anomaly_detection/test_each_view.py"

```
python test_each_view.py
```
Download embeddings (embeddings.zip) in [here](https://drive.google.com/file/d/1PMrQbx62T95SFkh1cBjbo7zXfQ8rsXkC/view?usp=sharing)

Our embedding files were extracted from undisclosed data. Therefore, the embeddedings file cannot be disclosed, but it is briefly disclosed. A password is required to decompress the file.

------
## Generate 3D anomaly map

Please run "Anomaly_detection/generate_3d_map.py"

```
python generate_3d_map.py
```

------
## Result

If you run four things sequentially, you will see that a "result_map" folder is created, storing the 3D anomaly map.

Result sample of VMPR-UAD 3D anomaly map .[[video](https://youtu.be/oerQdLnfPBQ)]

------

