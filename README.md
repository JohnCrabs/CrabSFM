# CrabSFM
A structure from motion algorithm.

To run this application open the main.py and insert the path to your folder of images. Also you can put the folder 
img2pointCloud to your project and import the crabSfm.py to your folder. Then run:

    CrabSFM(folderWithImagesPath, exportCloudFolder, fast=True/False)

to execute the process.

This is the first stable version of the code. By running the above command you can export 3 type of \*.ply files. The first 
is the point cloud created by the image pairs (each cloud unrelated to others), the second is point cloud after scale 
correction (all cloud has the same scaling system) and the last one is the related clouds (open all of them to take the full 
object).

In the next upload version I will add the last cloud type, which will sum all these cloud and create the full cloud. (I will 
also let the user have these cloud for data error checking, if a pair doesnt match with the others, then the final point 
cloud will be full of junk points.)

I will also create an efficient way to use the algorithm with multible dataset (opening a folder with subfolders and create a 
point cloud from every object in each folder - the output will be inside the folder).

### Some useful information about the algorith.
The sfm pipeline I used is the follow:
1) Open all images in folder (for saving memory, read only the path and open the images when needed)
2) Sort the paths (this means its better to use names like 000, 001, 002, ..., 00N and the images needs to be from left to   
right for the algorithm to run correctely.
3) Calculate feature points for each image (by default use AKAZE, but it can also find them with ORB, SIFT, SURF)
4a) Match only neighbor pairs (000-001, 001-002, ..., 00(N-1)-00N). If fast=True
4b) Match all images. (I think this gives better result, but it takes time to perform all calculations)
5) Find good matching points using the Lowes Ratio Distance Value. 
(Read this paper for more information: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
6) Find inlier from good points by calculating the fundamental matrix (cv2.findFundamental)
7) Approximate camera matrix. The camera matrix is:

                      | f_x   0   c_x |
        camera_mtrx = |  0   f_y  c_y |
                      |  0    0    1  |

## Additional Libraries
### video2images()
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

## rigid_transform_3D.py

A function solving the least squared method and calculate the Rotate(R) and Translate(t) matrices from a pair of 
corresponding poins. The pair needs to have the same scale. 

To be more specific the transformation equation is:

    X2 = s*R*X1 + t

In the above equation **X1** and **X2** describes the point cloud, **s** is the scale factor, **R** is a 3x3 rotation matrix 
and **t** is a 1x3 translation matrix (we assume that we are in a 3D world, you can solve the same problem in a 2D system or 
in a n-D system). This equation has 1_scale + 3_rotates + 3_translates = 7_uknown_factors (you need at least 3 points to solve this problem)

If X1 and X2 have beed scaled before, then scale = 1 and the above equation simplified to:

    X2 = R*X1 + t

Which means 3_rotates + 3_translates = 6_uknown_factors (you need at least 2 points to solve the equation - each points gives 
3 equations).

I took this file from: https://github.com/nghiaho12/rigid_transform_3D
And I thank him because I would spend many hours in front of the screen trying to solve this.
