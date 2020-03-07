from video2images import *
#from img2pointCloud.img2pointCloud import *
#from img2pointCloud.imgBlock2pointCloud import *
from img2pointCloud.crabSfm import *
#from img2pointCloud.my_sfm import *

videoPath = "inputData/VideosInput/"
exportVideoFramesPath = "outputData/VideoFrames/"
imgPath = "inputData/CalibData/RedMi_Test/"
calibrationPath = "inputData/CalibData/CalibrationIMG_RedMi/"
exportCalibrationParameters = "outputData/CalibrationParameters/"
exportDisparityMaps = "outputData/DisparityMaps/"
exportPointCloud = "outputData/PointCloud/"
testPath = "inputData/CalibData/test_sample/"

chessboardDim = (11, 8)

#videoINfolder2image(videoPath, exportVideoFramesPath, scaleFps=2, imgFormat="jpg")

#success = img2pcl(exportVideoFramesPath, exportPointCloud, exportDisparityMapsFolderPath=exportDisparityMaps)
#success = img2pcl(imgPath, exportPointCloud, focalLength=FOCAL_CALCULATE, imgCalibrationFolderPath=calibrationPath,
#                  chessboardDimensions=chessboardDim, exportCalibrationDataFolderPath=exportCalibrationParameters,
#                  exportDisparityMapsFolderPath=exportDisparityMaps)
#success = img2pcl(imgPath, exportPointCloud, focalLength=FOCAL_APPROXIMATE, imgCalibrationFolderPath=calibrationPath,
#                  chessboardDimensions=chessboardDim, exportCalibrationDataFolderPath=exportCalibrationParameters,
#                  exportDisparityMapsFolderPath=exportDisparityMaps)
#success = img2pcl(exportVideoFramesPath, exportPointCloud, focalLength=FOCAL_APPROXIMATE,
#                  imgCalibrationFolderPath=calibrationPath, chessboardDimensions=chessboardDim,
#                  exportCalibrationDataFolderPath=exportCalibrationParameters,
#                  exportDisparityMapsFolderPath=exportDisparityMaps, dpi=1200)

#success = imgBlock2PointCloud(exportVideoFramesPath, exportPointCloud, exportDisparityMaps)
#success = imgBlock2PointCloud(imgPath, exportPointCloud, exportDisparityMaps)

#success = CrabSFM(testPath, exportPointCloud)
#success = CrabSFM(calibrationPath, exportPointCloud)
#success = CrabSFM(imgPath, exportPointCloud)
success = CrabSFM(exportVideoFramesPath, exportPointCloud)

#success = run_Sfm(testPath, exportPointCloud)
#success = run_Sfm(calibrationPath, exportPointCloud)
#success = run_Sfm(imgPath, exportPointCloud)
#success = run_Sfm(exportVideoFramesPath, exportPointCloud)
