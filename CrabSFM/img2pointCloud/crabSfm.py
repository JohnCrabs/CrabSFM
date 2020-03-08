import os  # Use it for reading the paths
import cv2 as cv  # Key library for the calibration algorithm
import numpy as np  # Use it to save the parameters
import datetime as dt  # Use it for printing messages

import math as mth

imgFileFormats = (".jpg", ".jpeg", ".png", ".tiff")

H_MIN_SIZE = 2048
W_MIN_SIZE = 2048

AKAZE_METHOD = 0
ORB_METHOD = 1
SIFT_METHOD = 2
SURF_METHOD = 3

LOWE_RATIO = 0.9
INLIER_RATIO = 0.5
POSE_RATIO = 0.05


class Point3d:
    x: float
    y: float
    z: float

    def set_point(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Color:
    r: int
    g: int
    b: int

    def set_color(self, r: int, g: int, b: int):
        self.r = r
        self.g = g
        self.b = b


class Camera:
    fx = 1.0
    fy = 1.0
    cx = 0.0
    cy = 0.0

    mtrx = []

    def set_camera_parameters(self, f_x: float, f_y: float, c_x: float, c_y: float):
        self.fx = f_x
        self.fy = f_y
        self.cx = c_x
        self.cy = c_y

    def approximate_focal_length(self, width, height):
        if width > height:  # Check id width > height
            w = width  # Set w = width
        else:  # else
            w = height  # w = height
        focal = (0.7 * w + w) / 2  # Approximate the focal length as the the average of (70% of w + 100% of w)
        return focal

    def approximate_camera_parameters(self, width: int, height: int):
        focal = self.approximate_focal_length(width, height)
        self.fx = focal
        self.fy = focal
        self.cx = width / 2
        self.cy = height / 2

    def set_camera_matrix(self):
        cam_mtrx = ([self.fx, 0, self.cx],
                    [0, self.fy, self.cy],
                    [0, 0, 1])
        self.mtrx = np.array(cam_mtrx)

    def camera_info(self):
        print("")
        print_message("Camera Matrix = ")
        print(self.mtrx)


class PoseMatrix:
    R: []
    t: []

    T_mtrx: []

    def set_starting_pose_matrix(self):
        T = [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]]
        self.T_mtrx = np.array(T)

    def take_R_and_t(self):
        R = self.T_mtrx[:3, :3]
        t = self.T_mtrx[:3, 3:]
        return R, t

    def setPoseMatrix_R_t(self, R, t):
        Rt = []
        Rt.append(R)
        Rt.append(t)
        Rt = np.concatenate(Rt, axis=1)

        poseMtrx = []
        poseMtrx.append(Rt)
        poseMtrx.append([[0.0, 0.0, 0.0, 1.0]])
        poseMtrx = np.concatenate(poseMtrx, axis=0)

        self.T_mtrx = np.array(poseMtrx)

    def set_pose_mtrx_using_pair(self, pair_pose_mtrx):
        #print(pair_pose_mtrx)  # Uncomment for debugging
        #print(self.T_mtrx)  # Uncomment for debugging
        p_mtrx = np.dot(pair_pose_mtrx, self.T_mtrx)
        self.T_mtrx = np.array(p_mtrx)


class ProjectionMatrix:
    P_mtrx = []

    def set_starting_projection_matrix(self, cam_mtrx):
        projectionMtrx = []
        zeroMtrx = [[0], [0], [0]]
        projectionMtrx.append(cam_mtrx)
        projectionMtrx.append(zeroMtrx)
        projectionMtrx = np.concatenate(projectionMtrx, axis=1)
        self.P_mtrx = np.array(projectionMtrx)

    def set_projection_matrix_from_pose(self, R, t, cam_mtrx):
        R_t = np.transpose(R)
        m_R_t_t = np.dot(-R_t, t)

        P_tmp = []
        P_tmp.append(R_t)
        P_tmp.append(m_R_t_t)
        P_tmp = np.concatenate(P_tmp, axis=1)
        # print(P_tmp)

        P = np.dot(cam_mtrx, P_tmp)
        # print(P)
        self.P_mtrx = P


class Image:
    """
    This class contains all the variables which describes an Image. Also it contains the needed functions for
    setting values to these variables.
    """
    id: int  # Primary key (same as the index in the table)
    src: str  # Image relative or absolute path
    name: str  # Image name (without type)
    type_f: str  # Image type (for example .jpg)
    width: int  # Image width (in this algorithm the width of the downsample)
    height: int  # Image height (in this algorithm the height of the downsample)
    bands: int  # Image color bands

    keypoints = []  # The list of feature key points
    descriptors = []  # The list of descriptors for each key point
    kp_ids = []

    T_mtrx = PoseMatrix()  # Pose matrix for the current image
    P_mtrx = ProjectionMatrix()  # Projection matrix  for the current image

    def set_image(self, index: int, src: str, name: str, type_f: str, width: int, height: int, bands: int):
        self.id = index
        self.src = src
        self.name = name
        self.type_f = type_f
        self.width = width
        self.height = height
        self.bands = bands

    def set_feature_points(self, kp, desc):
        self.keypoints = kp
        self.descriptors = desc
        kp_id_tmp = []
        for i in range(0, len(kp)):
            kp_id_tmp.append(i)
        self.kp_ids = kp_id_tmp

    def image_info(self):
        print("")
        print_message("Information about image %s:" % self.name)
        print("Id: ", self.id)
        print("Path: ", self.src)
        print("Name: ", self.name)
        print("Type: ", self.type_f)
        print("(w, h, b): (%d" % self.width + ",%d" % self.height + ",%d" % self.bands + ")")

    def keypoint_info(self):
        message = " Img_%s_keypoints = " % self.name + "%d" % len(self.keypoints)
        print_message(message)

    def set_starting_pose_matrix(self, pose_mtrx: PoseMatrix):
        self.T_mtrx = pose_mtrx

    def append_pose_mtrx(self, pose_mtrx: PoseMatrix):
        self.T_mtrx.append(pose_mtrx)

    def set_starting_projection_matrix(self, proj_mtrx: ProjectionMatrix):
        self.P_mtrx = proj_mtrx


class MatchImages:
    match_id: int
    img_L_id = int
    img_R_id = int
    f_pts = []
    f_pts_L = []
    f_pts_R = []
    f_pts_indexes_L = []
    f_pts_indexes_R = []
    colors = []
    is_good = False

    def set_match(self, m_id: int, imgL_id: int, imgR_id: int, g_matches: [], g_matches_L: [], g_matches_R: [],
                  g_matches_id_L: [], g_matches_id_R: [], colors: [], is_good: bool):
        self.match_id = m_id
        self.img_L_id = imgL_id
        self.img_R_id = imgR_id
        self.f_pts = g_matches
        self.f_pts_L = g_matches_L
        self.f_pts_R = g_matches_R
        self.f_pts_indexes_L = g_matches_id_L
        self.f_pts_indexes_R = g_matches_id_R
        self.colors = colors
        self.is_good = is_good

    def create_index_table(self):
        index_tab = []
        for id_index in range(0, len(self.f_pts_indexes_L)):
            index_tab_tmp = []
            id_L = self.f_pts_indexes_L[id_index]
            id_R = self.f_pts_indexes_R[id_index]
            index_tab_tmp.append(id_L)
            index_tab_tmp.append(id_R)
            index_tab.append(index_tab_tmp)
        index_tab = np.array(index_tab)
        return index_tab

    def export_id_csv(self, path: str):
        id_list = []
        for i in range(0, len(self.f_pts_indexes_L)):
            tmp = []

            tmp.append(int(self.f_pts_indexes_L[i]))
            tmp.append(int(self.f_pts_indexes_R[i]))
            id_list.append(tmp)
        #print(id_list)
        np.savetxt(path, id_list, delimiter=",", fmt='%d')


class Landmark:
    l_id: int
    pnt3d: Point3d
    color: Color
    seen = 0

    match_id_list = []

    def set_landmark(self, l_id: int, x: float, y: float, z: float, seen: int, r=0, g=0, b=0):
        self.l_id = l_id
        pnt = Point3d()
        pnt.set_point(x, y, z)
        self.pnt3d = pnt
        col = Color()
        col.set_color(r, g, b)
        self.color = col
        self.seen = seen

    def append_pnt3d(self, pnt: Point3d):
        self.pnt3d.x += pnt.x
        self.pnt3d.y += pnt.y
        self.pnt3d.z += pnt.z
        self.seen += 1

    def finalize_coordinates(self):
        self.pnt3d.x /= self.seen
        self.pnt3d.y /= self.seen
        self.pnt3d.z /= self.seen

    def set_match_id_list(self, img_index_L, img_index_R):
        id_list = []
        id_list.append(img_index_L)
        id_list.append(img_index_R)
        self.match_id_list = id_list


class PairModel:
    id = 0
    imgL_id = 0
    imgR_id = 0
    points = []
    colors = []
    id_L_R_list = []

    def set_model(self, index: int, id_list: [], points: [], colors: []):
        self.id = index
        self.id_L_R_list = id_list
        self.points = points
        self.colors = colors


class BlockImage:
    images = []
    matches = []
    landmark = []
    pair_model = []
    l_block_model = []
    camera = Camera()

    block_match_list = []

    def append_image(self, img: Image):
        self.images.append(img)

    def info_for_images(self):
        for img in self.images:
            img.image_info()

    def append_matches(self, match: MatchImages):
        self.matches.append(match)

    def find_features(self, matchingMethod=AKAZE_METHOD):
        print("")
        print_message("Find Features for each Image.")
        for img_index in range(0, len(self.images)):  # For each image in block
            img = self.images[img_index]
            img_open = cv.imread(img.src, flags=cv.IMREAD_GRAYSCALE)  # Read the image
            img_size_tmp = img_open.shape  # Take the shape of image
            if len(img_size_tmp) is not 3:  # If image is dray scale set the bands to 1
                img_size_tmp = [img_size_tmp[0], img_size_tmp[1], 1]

            img_size = {"w": img_size_tmp[1], "h": img_size_tmp[0], "b": img_size_tmp[2]}  # Create Size instance

            img_open, img_size = imgDownsample(img_open, img_size["w"], img_size["h"])

            # Create key-point finder method checking the matchingMethod parameter (set by user)
            if matchingMethod is AKAZE_METHOD:
                method = cv.AKAZE_create()  # akaze method
            elif matchingMethod is ORB_METHOD:
                method = cv.ORB_create()  # orb method
            elif matchingMethod is SIFT_METHOD:
                method = cv.xfeatures2d.SIFT_create()  # sift method
            elif matchingMethod is SURF_METHOD:
                method = cv.xfeatures2d.SURF_create()  # surf method
            else:
                method = cv.AKAZE_create()  # if method checking failed use akaze method

            kp, descr = method.detectAndCompute(img_open, None)  # detect and compute keypoints

            img.set_feature_points(kp, descr)  # set key-points and descriptor per image

    def feature_info(self):
        for img in self.images:
            img.keypoint_info()

    def match_images_fast(self):
        print("")
        print_message("Feature Matching:")

        # Create matcher
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        # Find the Number of feature matching
        matchSize = 0
        block_size = len(self.images)
        for i in range(1, block_size):
            matchSize += 1

        print_message("Needs to perform %d matches." % matchSize)

        # Create matches
        matchCounter = 1
        for imgL_index in range(0, block_size-1):
            self.match_pairs(matcher, imgL_index, imgL_index+1, matchCounter, matchSize)
            matchCounter += 1  # increase the matchCounter

        self.create_block_match_list()

    def match_images(self):
        print("")
        print_message("Feature Matching:")

        # Create matcher
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)

        # Find the Number of feature matching
        matchSize = 0
        block_size = len(self.images)
        for i in range(1, block_size):
            matchSize += block_size - i

        print_message("Needs to perform %d matches." % matchSize)

        # Create matches
        matchCounter = 1
        for imgL_index in range(0, block_size-1):
            for imgR_index in range(imgL_index+1, block_size):
                self.match_pairs(matcher, imgL_index, imgR_index, matchCounter, matchSize)
                matchCounter += 1  # increase the matchCounter

        for m in self.matches:
            imgL_name = self.images[m.img_L_id].name
            imgR_name = self.images[m.img_R_id].name
            exportName = "./outputData/csv/" + imgL_name + "_" + imgR_name + ".csv"
            if not os.path.exists("./outputData/csv/"):
                os.makedirs("./outputData/csv/")
            m.export_id_csv(exportName)

        self.create_block_match_list()

    def match_pairs(self, matcher, imgL_index: int, imgR_index: int, matchCounter: int, matchSize: int):
        img_L = self.images[imgL_index]
        img_R = self.images[imgR_index]
        print("")
        message = "Match %d" % matchCounter + " out of " + "%d matches needed to perform." % matchSize
        print_message(message)

        img_L_name = img_L.name
        img_R_name = img_R.name
        message = "Match images %s" % img_L_name + " and " + "%s" % img_R_name
        print_message(message)

        kp_L = self.images[imgL_index].keypoints
        kp_R = self.images[imgR_index].keypoints

        desc_L = self.images[imgL_index].descriptors
        desc_R = self.images[imgR_index].descriptors

        kp_id_list_L = self.images[imgL_index].kp_ids
        kp_id_list_R = self.images[imgR_index].kp_ids

        matched_points = matcher.knnMatch(desc_L, desc_R, k=2)  # Run matcher

        # Find all good points as per Lower's ratio.
        good_matches = []
        points_L_img = []
        points_R_img = []
        points_L_img_ids = []
        points_R_img_ids = []
        match_pnt_size = 0
        for m, n in matched_points:
            match_pnt_size += 1
            if m.distance < LOWE_RATIO * n.distance:
                good_matches.append(m)
                points_L_img.append(kp_L[m.queryIdx].pt)  # Take p_coords for left img
                points_R_img.append(kp_R[m.trainIdx].pt)  # Take p_coords for right img

                points_L_img_ids.append(kp_id_list_L[m.queryIdx])  # Take the ids for the left image
                points_R_img_ids.append(kp_id_list_R[m.trainIdx])  # Take the ids for the right image

        g_pnts_size = len(good_matches)
        message = "Found %d" % g_pnts_size + " good matches out of %d" % match_pnt_size + " matching points."
        print_message(message)

        good_matches = np.array(good_matches)
        points_L_img = np.array(points_L_img)
        points_R_img = np.array(points_R_img)
        points_L_img_ids = np.array(points_L_img_ids)  # ID_L
        points_R_img_ids = np.array(points_R_img_ids)  # ID_R

        #print(points_L_img_ids)
        #print(points_R_img_ids)

        # Calculate inliers using Fundamental Matrix
        print_message("Calculate inlier matches.")
        pts_L_fund = np.int32(points_L_img)  # Transform float to int32
        pts_R_fund = np.int32(points_R_img)  # Transform float to int32

        F, mask = cv.findFundamentalMat(pts_L_fund, pts_R_fund)  # Find fundamental matrix using RANSARC
        # We select only inlier points
        pts_inlier_matches = good_matches[mask.ravel() == 1]
        pts_inlier_L = points_L_img[mask.ravel() == 1]  # Select inliers from imgL using fundamental mask
        pts_inlier_R = points_R_img[mask.ravel() == 1]  # Select inliers from imgR using fundamental mask
        pts_inlier_L_ids = points_L_img_ids[mask.ravel() == 1]  # Select inlier IDS from imgL_index
        # using fundamental mask
        pts_inlier_R_ids = points_R_img_ids[mask.ravel() == 1]  # Select inliers IDS from imgR_index
        # using fundamental mask

        # Calculate the Color of an Image
        pts_L_fund = pts_L_fund[mask.ravel() == 1]
        color_inlier = find_color_list(img_L, pts_L_fund)
        # print(colors)

        match_tmp = MatchImages()  # Create a temporary match item
        # match_tmp.set_match(m_id=matchCounter-1, imgL_id=img_L.id, imgR_id=img_R.id, g_matches=good_matches,
        #                    g_matches_L=points_L_img, g_matches_R=points_R_img,
        #                    g_matches_id_L=points_L_img_ids, g_matches_id_R=points_R_img_ids)

        g_pnt_size = len(good_matches)  # Find the size of q_pnts
        inliers_size = len(pts_inlier_L_ids)  # Find the size of inliers

        match_is_good = True
        if inliers_size < INLIER_RATIO * g_pnts_size:
            match_is_good = False

        match_tmp.set_match(m_id=matchCounter - 1, imgL_id=img_L.id, imgR_id=img_R.id,
                            g_matches=pts_inlier_matches, g_matches_L=pts_inlier_L, g_matches_R=pts_inlier_R,
                            g_matches_id_L=pts_inlier_L_ids, g_matches_id_R=pts_inlier_R_ids,
                            colors=color_inlier, is_good=match_is_good)

        # Print the sizes to screen so we can see the difference.
        # In every step we need to exclude unfitting points for creating better results.
        message = "Found %d" % inliers_size + " inlier matches out of %d" % g_pnt_size + \
                  " good feature matching points."
        print_message(message)

        self.matches.append(match_tmp)

    def set_camera(self):
        print("")
        print_message("Approximate Camera Matrix")
        cam = Camera()
        cam.approximate_camera_parameters(self.images[0].width, self.images[0].height)
        cam.set_camera_matrix()
        self.camera = cam
        self.camera.camera_info()

    def find_landmarks(self, exportPath):
        print("")
        print_message("Find Landmarks")

        matchSize = len(self.matches)
        #print(matchSize) # Uncomment for debugging

        cam_mtrx = self.camera.mtrx

        # Landmark Founder
        matchCounter = 1
        set_start_mtrx = True
        landmarkCounter = 0
        pairModelCounter = 0
        for match in self.matches:  # for each matching pair

            imgL_index = match.img_L_id  # read left img id
            imgR_index = match.img_R_id  # read right img id

            imgL = self.images[imgL_index]  # read left img (we need it for the index key table)
            imgR = self.images[imgR_index]  # read right img (we need it for general information like name)

            #print(imgR)

            imgL_name = imgL.name  # read left img name
            imgR_name = imgR.name  # read right img name

            pts_inlier_L = match.f_pts_L  # read good matching points left
            pts_inlier_R = match.f_pts_R  # read good matching points right

            pts_inlier_L_id = match.f_pts_indexes_L  # read good matching points left indexes
            pts_inlier_R_id = match.f_pts_indexes_R  # read good matching points right indexes

            colors = match.colors

            print("")
            message = "(%d / " % matchCounter + "%d)" % matchSize
            print_message(message)
            message = "Find landmark in images %s" % imgL_name + " and %s." % imgR_name
            print_message(message)

            # Calculate Essential Matrix
            # Uncomment for compare or testing
            # E, mask = cv.findEssentialMat(g_pnts_L, g_pnts_R, cam_mtrx)
            # print(E)

            # I prefer inlier solution.
            E, mask = cv.findEssentialMat(pts_inlier_L, pts_inlier_R, cam_mtrx)
            #print(E)

            # Calculate pose matrix R and t
            # poseVal = The number of pose points (we'll use these points to create the cloud)
            #    R    = Rotate Matrix
            #    t    = Translate Matrix
            #  mask   = Take values 0 and 1 and if 1 this point is pose point (object point)
            poseVal, R, t, mask = cv.recoverPose(E, pts_inlier_L, pts_inlier_R, cam_mtrx)
            poseMask = mask  # Keep the mask in a variable poseMask (I done this for easier code reading)

            # The poseVal value indicates the candidate number of new object points.
            # I named it candidate because some of these points may be visible from previous images.
            # In this case we need to find the average of these points. This method remove dublicate points and
            # increase the accuracy of the final point cloud.
            g_p_size = len(pts_inlier_L_id)
            message = "Found %d" % poseVal + " candidate object points out %d suggested matching points." % g_p_size
            print_message(message)

            if set_start_mtrx:
                set_start_mtrx = False
                pose_mtrx_img_0 = PoseMatrix()
                pose_mtrx_img_0.set_starting_pose_matrix()
                imgL.set_starting_pose_matrix(pose_mtrx_img_0)

                proj_mtrx_img_0 = ProjectionMatrix()
                proj_mtrx_img_0.set_starting_projection_matrix(self.camera.mtrx)
                imgL.set_starting_projection_matrix(proj_mtrx_img_0)

            landmark_debugging_list = []
            if poseVal > POSE_RATIO * g_p_size and match.is_good:
                # Create the pose matrices.
                # Create the Pose and Projection Matrices
                print_message("Calculate Pose Matrices:")

                pose_mtrx_L_T = imgL.T_mtrx.T_mtrx   # imgL.T_mtrx is a list of all pose matrices of this image
                                                     # imgL.T_mtrx[0].T_mtrx is always the matrix of the left img
                pose_mtrx_R = PoseMatrix()
                pose_mtrx_R.setPoseMatrix_R_t(R, t)
                pose_mtrx_R.set_pose_mtrx_using_pair(pose_mtrx_L_T)

                proj_mtrx_L_P = imgL.P_mtrx.P_mtrx

                R, t = pose_mtrx_R.take_R_and_t()
                proj_mtrx_R_P = ProjectionMatrix()
                proj_mtrx_R_P.set_projection_matrix_from_pose(R, t, self.camera.mtrx)

                if imgR.id - imgL.id is 1:
                    imgR.set_starting_pose_matrix(pose_mtrx_R)
                    imgR.set_starting_projection_matrix(proj_mtrx_R_P)

                #print("")
                #print("pose_mtrx_L = \n", pose_mtrx_L_T)  # Uncomment for debug
                #print("")
                #print("pose_mtrx_R = \n", pose_mtrx_R.T_mtrx)  # Uncomment for debug
                #print("")
                #print("proj_mtrx_L = \n", proj_mtrx_L_P)  # Uncomment for debug
                #print("")
                #print("proj_mtrx_R = \n", proj_mtrx_R_P.P_mtrx)  # Uncomment for debug

                # Triangulate
                proj_mtrx_R_P = proj_mtrx_R_P.P_mtrx
                print_message("Triangulation.")

                triang_pnts_L = np.transpose(pts_inlier_L)
                triang_pnts_R = np.transpose(pts_inlier_R)

                points4D = cv.triangulatePoints(projMatr1=proj_mtrx_L_P,
                                                projMatr2=proj_mtrx_R_P,
                                                projPoints1=triang_pnts_L,
                                                projPoints2=triang_pnts_R)
                #print(points4D)  # Uncomment for debugging

                # Find Good LandMark Points and Set Them to List
                #print(pts_inlier_L_id)
                #print(pts_inlier_R_id)
                for l_index in range(0, g_p_size):
                    if poseMask[l_index] != 0:
                        #print(poseMask[l_index]) # Uncomment for debugging
                        #print(l_index)  # Uncomment for debugging

                        pt3d = Point3d()

                        pt3d.x = points4D[0][l_index] / points4D[3][l_index]
                        pt3d.y = points4D[1][l_index] / points4D[3][l_index]
                        pt3d.z = points4D[2][l_index] / points4D[3][l_index]

                        # OpenCV images are in BGR system so b=0, g=1, r=2
                        r = colors[l_index][0]
                        g = colors[l_index][1]
                        b = colors[l_index][2]
                        # print(r, g, b)
                        l_pnt = Landmark()
                        l_pnt.set_landmark(landmarkCounter, pt3d.x, pt3d.y, pt3d.z, 1, r, g, b)
                        l_pnt.set_match_id_list(pts_inlier_L_id[l_index], pts_inlier_R_id[l_index])

                        landmark_debugging_list.append(l_pnt)
                        landmarkCounter += 1
                pair_model_tmp = PairModel()
                exportName = exportPath + imgL_name + "_" + imgR_name + ".ply"
                exp_points, exp_colors, exp_id = transform_landmark_to_list_items(landmark_debugging_list)
                message = "Export Pair Model as : " + exportName
                print_message(message)
                export_as_ply(exp_points, exp_colors, exportName)

                pair_model_tmp.set_model(pairModelCounter, exp_id, exp_points, exp_colors)
                self.pair_model.append(pair_model_tmp)
            else:
                message = "Cannot create pair model from images " + imgL_name + " and " + imgR_name + \
                          ", due to few points."
                print_message(message)

                imgR.set_starting_pose_matrix(imgL.T_mtrx)
                imgR.set_starting_projection_matrix(imgL.P_mtrx)
            matchCounter += 1  # increase the matchCounter

    def finalize_landmark(self):
        for landmark in self.landmark:
            landmark.finalize_coordinates()

    def transform_landmark_to_list(self):
        points = []
        colors = []
        id_list = []
        for l_pnt in self.landmark:
            x = l_pnt.pnt3d.x
            y = l_pnt.pnt3d.y
            z = l_pnt.pnt3d.z
            r = l_pnt.color.r
            g = l_pnt.color.g
            b = l_pnt.color.b
            pt_tmp = [x, y, z]
            #col = [0, 0, 0]
            col = [r, g, b]
            #print(col)
            points.append(pt_tmp)
            colors.append(col)
            id_list.append(l_pnt.match_id_list)
            #print(pt_tmp)

        points = np.array(points)
        colors = np.array(colors)

        return points, colors, id_list

    def create_block_match_list(self):
        block_match_list_tmp = []

        for img_id in range(0, len(self.images)-1):
            img = self.images[img_id]
            kp_ids = img.keypoints

            match_list_tmp = []
            for i in range(0, len(kp_ids)):
                tmp = [i]
                for j in range(img.id+1, len(self.images)):
                    tmp.append(-1)

                match_list_tmp.append(tmp)
            #print(match_list_tmp)
            block_match_list_tmp.append(match_list_tmp)

        for m in self.matches:
            imgL_id = m.img_L_id
            imgR_id = m.img_R_id
            #print(imgL_id, imgR_id)
            #print(block_match_list_tmp[imgL_id])
            match_ids_L = m.f_pts_indexes_L
            match_ids_R = m.f_pts_indexes_R
            for index in range(0, len(match_ids_L)):
                block_match_list_tmp[imgL_id][match_ids_L[index]][imgR_id-imgL_id] = match_ids_R[index]
            #print(block_match_list_tmp[imgL_id])
        self.block_match_list = block_match_list_tmp
        print(self.block_match_list)

# -------------------------------------------------------------- #
#
# -------------------------------------------------------------- #


def CrabSFM(src: str, exportCloud: str, fast=True):
    """
    This function read all images in folder src and run the sfm pipeline to create a point cloud (model) end export
    it an *.ply file in the exportCloud path.
    :param fast:
    :param src: The relative or absolute path to the folder
    :param exportCloud: The relative or absolute path to the export folder/file
    :return: True when the process finished
    """
    block = open_Images_in_Folder(src=src)
    block.info_for_images()
    block.set_camera()
    block.find_features()
    block.feature_info()
    if fast is True:
        block.match_images_fast()
    else:
        block.match_images()
    block.find_landmarks(exportCloud)
    #print("")
    #print(len(block.pair_model))
    return True

# -------------------------------------------------------------- #
#
# -------------------------------------------------------------- #


def print_message(message: str):
    print(str(dt.datetime.now()) + " : " + message)

# -------------------------------------------------------------- #
#
# -------------------------------------------------------------- #


def downsample(image, scaleFactor):
    for i in range(0, scaleFactor - 1):
        # Check if image is color or grayscale
        if len(image.shape) > 2:
            row, col = image.shape[:2]
        else:
            row, col = image.shape
        image = cv.pyrDown(image, dstsize=(col // 2, row // 2))
    return image


def imgDownsample(image, width: int, height: int, showMessages=False):

    width_cpy = width  # Create a copy from width
    height_cpy = height  # Create a copy from height

    # If show messages is True, print message
    if showMessages:
        print_message("Downsample message.")
    image_downsample = image

    # Check if width is greater than height
    if width_cpy > height_cpy:
        if width_cpy > W_MIN_SIZE:
            scaleFactor = int(width_cpy / W_MIN_SIZE)
            if showMessages:
                message = "Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)

    # Check if height is greater than width
    elif width_cpy < height_cpy:
        if height_cpy > H_MIN_SIZE:
            scaleFactor = int(height_cpy / H_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)

    # If previous checks failed then image sizes are the same.
    # This branch is needed cause the program must check if the
    #      sizes are greater than default algorithm sizes.
    else:
        if width_cpy > W_MIN_SIZE:
            scaleFactor = int(width_cpy / W_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)
        elif height_cpy > H_MIN_SIZE:
            scaleFactor = int(height_cpy / H_MIN_SIZE)
            if showMessages:
                message = " : Downsample Scale Factor = %d." % scaleFactor
                print_message(message)
            image_downsample = downsample(image, scaleFactor)

    img_new_size = image_downsample.shape  # Take the shape of image
    if len(img_new_size) is not 3:  # If image is dray scale set the bands to 1
        img_new_size = [img_new_size[0], img_new_size[1], 1]
    img_new_size = [img_new_size[1], img_new_size[0], 1]

    return image_downsample, img_new_size

# -------------------------------------------------------------- #
#
# -------------------------------------------------------------- #


def open_Images_in_Folder(src: str):
    """
    This function read all the img paths in a given folder.
    :param src: The path to the folder.
    :return: The list of all the path.
    """
    block = BlockImage()  # Create BlockImage() item

    print("\n")
    print_message("Reading files in folder.")
    imgPathList = []  # A list with the relative path of images

    # Create a List with the path of Images
    for r, d, f in os.walk(src):
        for imgFormat in imgFileFormats:
            for file in f:
                if imgFormat in file:
                    imgPathList.append(os.path.join(r, file))
    imgPathList.sort()  # Sort the list

    print("")
    print_message("Found images:")
    for path in imgPathList:
        print(path)

    counter_id = 0
    print("")
    for file in imgPathList:  # For each file in path list
        img_name_type = os.path.basename(file)  # Read the name.type
        img_name = os.path.splitext(img_name_type)[0]  # Take the name
        img_type = os.path.splitext(img_name_type)[1]  # Take the type

        img_open = cv.imread(file)  # Read the image
        if img_open.size == 0:  # Error Checking
            message = "Error: Cannot open image at %s" % file
            print_message(message)
        else:  # If image opened
            message = "Read image file at : %s" % file
            print_message(message)
            img_size_tmp = img_open.shape  # Take the shape of image
            if len(img_size_tmp) is not 3:  # If image is dray scale set the bands to 1
                img_size_tmp = [img_size_tmp[0], img_size_tmp[1], 1]

            img_size = {"w": img_size_tmp[1], "h": img_size_tmp[0], "b": img_size_tmp[2]}  # Create Size instance

            img_open, img_size = imgDownsample(img_open, img_size["w"], img_size["h"], True)

            img = Image()  # Create Image() item
            img.set_image(index=counter_id, src=file, name=img_name, type_f=img_type,
                          width=img_size[0], height=img_size[1], bands=img_size[2])

            block.append_image(img)

            counter_id += 1
    return block


def export_as_ply(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    ply_header = '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')


def findMax_2x2(mtrx):
    x = 0
    y = 0

    for m in mtrx:
        if x < m[0]:
            x = m[0]
        if y < m[1]:
            y = m[1]
    return x, y


def find_color_list(img: Image, pts_inlier: []):
    colors = []
    img_open = cv.imread(img.src)
    img_size = img_open.shape
    img_open, img_size = imgDownsample(img_open, img_size[1], img_size[0])
    #img_draw_kp = cv.drawKeypoints(img_L_open, kp_match_L_inlier, None, color=(0, 255, 0), flags=0)
    # cv.imwrite("./img.jpg", img_draw_kp)

    blue = img_open[:, :, 0]
    green = img_open[:, :, 1]
    red = img_open[:, :, 2]
    # cv.imwrite("./blue.jpg", blue)
    # cv.imwrite("./green.jpg", green)
    # cv.imwrite("./red.jpg", red)
    # cv.imwrite("./img.jpg", img_L_open)
    # x, y = findMax_2x2(pts_L_fund)
    # print(x, y)
    for indx in pts_inlier:
        i_L = indx[1]
        j_L = indx[0]
        # print(i_L)
        # print(j_L)
        col_r = red[i_L][j_L]
        col_g = green[i_L][j_L]
        col_b = blue[i_L][j_L]
        col = [col_r, col_g, col_b]
        # col = [0, 150, 0]
        colors.append(col)
    return colors


def transform_landmark_to_list_items(landmark: []):
    points = []
    colors = []
    id_list = []
    for l_pnt in landmark:
        x = l_pnt.pnt3d.x
        y = l_pnt.pnt3d.y
        z = l_pnt.pnt3d.z
        r = l_pnt.color.r
        g = l_pnt.color.g
        b = l_pnt.color.b
        pt_tmp = [x, y, z]
        #col = [0, 0, 0]
        col = [r, g, b]
        #print(col)
        points.append(pt_tmp)
        colors.append(col)
        id_list.append(l_pnt.match_id_list)
        #print(pt_tmp)

    points = np.array(points)
    colors = np.array(colors)
    id_list = np.array(id_list)

    return points, colors, id_list
