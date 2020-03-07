# CrabSFM
A structure from motion algorithm.

To run this application open the main.py and insert the path to your folder of images. Also you can put the folder 
img2pointCloud to your project and import the crabSfm.py to your folder. Then run:

    CrabSFM(folderWithImagesPath, exportCloudFolder, fast=True/False)

to execute the process.

Currently thie version export good clouds only from a good pair of images. The next version will be able to create the submodels (pairs of point cloud) to a complete model created from all images.
