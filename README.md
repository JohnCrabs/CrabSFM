# CrabSFM
A structure from motion algorithm.

To run this application open the main.py and insert the path to your folder of images. Also you can put the folder 
img2pointCloud to your project and import the crabSfm.py to your folder. Then run:

    CrabSFM(folderWithImagesPath, exportCloudFolder, fast=True/False)

to execute the process.

Currently thie version export good clouds only from a good pair of images. The next version will be able to create the 
submodels (pairs of point cloud) to a complete model created from all images.

## video2images()
This library now can export frames drom mp4 video files. If you want to export some other video type frames, you just have to 
add the specific type to the video-type list and run the function as shown in main.py

Also by choosing diffFolder=True the algorith can now save the frames of each video in different folder. However it keeps the counter numbering in all folder. In the future I may add the option to restart the counter for the different folders.
