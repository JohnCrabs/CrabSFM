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

Also by choosing diffFolder=True the algorith can now save the frames of each video in different folder. You can also set if 
you want to keep the numbering or start over from 000 when changing folder by adding **restartNumbering=True/False**.

Example:
    
    videoINfolder2image(in_folder, out_folder, scaleFps=1, imgFormat="jpg", diffFolder=True, restartNumbering=True)
    
## visualizeCloud()
A simple cloud visualizer using **open3d** library. Currently you must give the path to the \*.ply file. In the future I'll 
add a simple command line so you can load multiple \*.ply files and visualize each of them by typing an index id at the 
command line. 
