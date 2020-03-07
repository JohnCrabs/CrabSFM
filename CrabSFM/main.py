from video2images import *
from img2pointCloud.crabSfm import *

videoPath = "inputData/VideosInput/"
exportVideoFramesPath = "outputData/VideoFrames/"
imgPath = "inputData/CalibData/RedMi_Test/"
calibrationPath = "inputData/CalibData/CalibrationIMG_RedMi/"
exportCalibrationParameters = "outputData/CalibrationParameters/"
exportDisparityMaps = "outputData/DisparityMaps/"
exportPointCloud = "outputData/PointCloud/"
testPath = "inputData/CalibData/test_sample/"

pathVideosAI = "inputData/videos_AI/"

#videoINfolder2image(videoPath, exportVideoFramesPath, scaleFps=2, imgFormat="jpg", diffFolder=False)
videoINfolder2image(pathVideosAI, exportVideoFramesPath, scaleFps=1, imgFormat="jpg", diffFolder=True)

#success = CrabSFM(testPath, exportPointCloud)
#success = CrabSFM(calibrationPath, exportPointCloud)
#success = CrabSFM(imgPath, exportPointCloud)
#success = CrabSFM(exportVideoFramesPath, exportPointCloud)
