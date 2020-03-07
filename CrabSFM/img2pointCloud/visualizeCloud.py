import os
from open3d import *


def showCloud(path: str):
    win_name = os.path.basename(path)

    cloud = openCloud(path)
    draw_geometries([cloud], window_name=win_name, width=740, height=680)


def openCloud(path: str):
    cloud = read_point_cloud(path)
    return cloud
