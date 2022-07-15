########################################################################################################################
# FILE WITH THE CLASS DETECTION
########################################################################################################################

# LIBRARIES TO IMPORT
import math
import numpy as np
import os
import joblib
import statistics
import cv2


class Detection:
    #################################################################
    # BUILDER OF THE CLASS
    #################################################################
    def __init__(self, file_ObjectClasses, version_YOLO):
        # CHECK THE YOLO VERSION
        self.coord_hor, self.coord_depth = self.check_vYOLO(version_YOLO)
        # LOAD THE OBJECT-CLASS LABELS OUR YOLO MODEL WAS TRAINED ON
        try:
            labelsPath = "Resources/" + file_ObjectClasses
            self.labels = open(labelsPath).read().strip().split("\n")
            self.n_classes = len(self.labels)  # Number of classes of objects
        except:
            print("File " + file_ObjectClasses + " is not found")

        # PARAMETERS FOR THE CAMERA
        self.aperture_cam = 2 * math.atan(2.88/5.765)   # Camera aperture for viewing angle (radians)
        self.aperture = self.aperture_cam / 2           # Half value of the aperture angle of view's camera (radians)

        # PARAMETERS FOR THE IMAGE CAPTURATED
        self.height = 480   # In pixel
        self.width = 640    # In pixel

        # VALUES OF NORMALIZATION FOR TRACKING
        self.norm_dist = 25            # Value in cm
        self.norm_ang = 10             # Value in grades (º)
        self.norm_x_img = self.width   # Value in pixel
        self.norm_y_img = self.height  # Value in pixel

        # objects_tracking: structure storing all the information of the tracked objects.
        # Each entry objects_tracking[n] include a dictionary for each tracked object within
        # the n-th class with the following information:
        # a) images in which appear the tracked object
        # b) their x,y image coordinates
        # c) estimated pose of objects
        self.objects_tracking = [None] * self.n_classes
        # ind_objects: list in which each entry ind_objects[n] stores at any time the number
        # of tracked objects within the n-th class
        self.ind_objects = [0] * self.n_classes

        # PREGUNTAR A SERGIO!!!
        self.displacement_previous = [None] * self.n_classes

        # Load the classifier
        self.model_svm = joblib.load('model_svmFilterGiros5_v2.pkl')

        ##########################################################################################
        # DECLARACIONES DE LOS NODOS, LOS TOPICS PUBLISHER Y SUBSCRIBER CUANDO SE INTEGRE EN ROS
        ##########################################################################################

        ##########################################################################################

    ###################################################################################################
    # FUNCTION FOR CHECKING THE VERSION OF YOLO
    ###################################################################################################

    def check_vYOLO(self, version):
        # In case to use the first version, the depth is considerate
        if version == 'YOLO_v1':
            return 2, 1
        # In otherwise, the depth is not considerate
        elif version == 'YOLO_v2':
            return 1, None

    ###################################################################################################
    # FUNCTION THAT READS THE OUTPUT DATA FROM YOLO
    ###################################################################################################

    def read_CoordYOLO(self, YOLO_BoundingBox, YOLO_depth=None):

        """
        This function read the output data created with the CNN YOLO. The data contain the object class
        and the estimation of the centroid coordinates (x,y) in pixel from the image.
        :param YOLO_BoundingBox: this is a .txt file with the coordinates of bounding box of objects
        :param YOLO_depth: this is a .txt with the data depth from YOLO
        :return:
        """
        # Init the var 'depth' to false. In this version is considerate the depth of objects from images
        depth = False

        # Obtain the horizontal coordinates from detected objects and their depth
        xcoord_object_detected = [None] * self.n_classes           # Coordinate horizontal from detected objects
        ycoord_object_detected = [None] * self.n_classes           # Coordinate vertical from detected objects
        depth_object_detected = [None] * self.n_classes            # Depth from the detected objects

        # Open and read the file where is saved the coordinates of bounding box from objects detected by YOLO
        f = open(YOLO_BoundingBox, 'r')

        # In case to have the depth of the object is considerate, the previous step is read the .txt file
        if YOLO_depth is not None:
            # Open the file
            f_depth = open(YOLO_depth, 'r')
            # Auxiliar vars: matrix depth and bounding depth
            matrix_depth = []
            # Loop for complete the matrix depth
            for d in f_depth:
                row = d.split(' ')
                # Steps for delete the '\n' from last value from list
                resultado = row[-1]
                resultado = resultado[0:4]
                row[-1] = resultado
                # Append the results
                matrix_depth.append(row)

            matrix_depth = np.array(matrix_depth)
            matrix_depth = matrix_depth.astype(float)

        # Loop for saved in the vars the data
        for x in f:
            line = x.split(' ')
            # Centroid of objects
            xc = int(float(line[2]) + 0.5 * (float(line[3]) - float(line[2])))
            yc = int(float(line[4]) + 0.5 * (float(line[5]) - float(line[4])))

            # In case to have the depth of the object is considerate
            if YOLO_depth is not None:
                depth = True
                xb1 = int(float(line[2]))
                xb2 = int(float(line[3]))
                yb1 = int(float(line[4]))
                yb2 = int(float(line[5]))
                # Obtain the measures of depth from bounding box
                bounding_depth = matrix_depth[yb1:yb2, xb1:xb2]
                bounding_depth = np.reshape(bounding_depth, (1, np.size(bounding_depth)))
                measure_depth = statistics.median(bounding_depth[0, :])

            for obj in range(self.n_classes):
                # Case to detect an object with the object class
                if int(line[0]) is obj:

                    if xcoord_object_detected[obj] is None:
                        # Create an empty list
                        xcoord_object_detected[obj] = []
                    # Write in the list the horizontal coordinate value
                    xcoord_object_detected[obj].append(xc)
                    if ycoord_object_detected[obj] is None:
                        # Create an empty list
                        ycoord_object_detected[obj] = []
                    # Write in the list the vertical coordinate value
                    ycoord_object_detected[obj].append(yc)

                    if depth:
                        if depth_object_detected[obj] is None:
                            # Create an empty list
                            depth_object_detected[obj] = []
                            # Write in the list the depth value
                            # depth_object_detected[obj].append(float(line[self.coord_depth]))
                            depth_object_detected[obj].append(measure_depth)
                        else:
                            # Write in the list the depth value
                            # depth_object_detected[obj].append(float(line[self.coord_depth]))
                            depth_object_detected[obj].append(measure_depth)

                if xcoord_object_detected[obj] is not None:
                    # Sort the list from least to greatest
                    xcoord_object_detected[obj] = sorted(xcoord_object_detected[obj])

                if ycoord_object_detected[obj] is not None:
                    # Sort the list from least to greatest
                    ycoord_object_detected[obj] = sorted(ycoord_object_detected[obj])

                if depth and depth_object_detected[obj] is not None:
                    # Sort the list from least to greatest
                    depth_object_detected[obj] = sorted(depth_object_detected[obj])

        return xcoord_object_detected, ycoord_object_detected, depth_object_detected

    ##################################################################################################
    # FUNCTIONS OF GEOMETRY: CREATE THE LINE BETWEEN TWO POINTS AND KNOW THE INTERSECTION BETWEEN
    # TWO LINES
    ##################################################################################################

    def line(self, p1, p2):

        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(self, L1, L2):

        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return x, y
        else:
            return None, None  # return False

    def line_intersection(self, line1, line2):

        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    ####################################################################################################
    # FUNCTION FOR ESTIMATE THE POSE
    ####################################################################################################

    def estimation_pose_object(self, x_coord, depth, inc_y):
        """
        Function to calculate the estimation pose of the object using the intersection between two
        lines.

        :param x_coord: coordinate 'x' of the object in the image
        :param depth: parameter of depth from object in meters
        :param inc_y: current displacement with the wheelchair in y in cm
        :return: the intersection between the line and circumference
        """
        # Angle's detection of the object in the previous image
        angle = math.atan(((x_coord-self.width/2) * math.tan(self.aperture))/(self.width/2))

        # Calculate the point of circumference using the depth data (radio) and angle
        R_y = (depth * 100) * math.cos(angle) + inc_y
        R_x = (depth * 100) * math.sin(angle)

        return R_x, R_y

    ####################################################################################################################
    # BLOCK OF FUNCTIONS FOR EVALUATING THE TRACKING OBJECTS IN A SEQUENCE
    ####################################################################################################################

    ####################################################################################################################
    # FUNCTION FOR CALCULATE THE MATRIX OF PROBABILITIES FROM THE TRACKING OBJECTS, THIS FUNCTION IS USED IN THE MAIN
    # FUNCTION FOR EVALUATING A SEQUENCE OF IMAGES
    ####################################################################################################################

    def Tracking_Matrix_Probability(self, cent1_objects, cent2_objects, objs, pos):
        """
        This function calculate the matrix of probability from the tracking objects in each pair of
        images from the sequence
        :param cent1_objects: is a list with x,y centroid of the objects from the first image (previous image)
        :param cent2_objects: is a list with x,y centroid of the objects from the second image (actual image)
        :param objs: list that contains the list that store the indexes of objects from the first and
                     second image (previous and actual image)
        :param pos: list with the info of increments in pose x,y,theta from the wheelchair
        :return matrix_tracking_prob: return the matrix
        """
        # Empty List for save al matrix probability for each class
        matrix_tracking_prob = [None] * self.n_classes
        # Loop for calculate the matrix
        for n in range(self.n_classes):
            # With this condition evaluate all objects except the detection of people
            if self.labels[n] != 'person':
                # Check if there are detected objects of the specific class in
                # both images.
                if cent1_objects[0][n] is not None and cent2_objects[0][n] is not None:

                    # Matrix of tracking_probabilities. Each entry (i,j) indicates the output probability
                    # of SVM between the objet i of image 1 and object j of the image 2.
                    # If there is no correspondence according to SVM, the (i,j) entry is set to 0.
                    matrix_tracking_prob_act = np.zeros(shape=(len(cent1_objects[0][n]),
                                                               len(cent2_objects[0][n])))

                    # Determine matrix of probabilities of tracking correspondences.
                    for i in range(len(cent1_objects[0][n])):
                        for j in range(len(cent2_objects[0][n])):
                            # Analysis of tracking-correspondences
                            xc1 = cent1_objects[0][n][i]
                            yc1 = cent1_objects[1][n][i]
                            xc2 = cent2_objects[0][n][j]
                            yc2 = cent2_objects[1][n][j]

                            if objs[0][n][i] == 0:
                                displacement_vector = [0, 0]
                            else:
                                # print('DisplacementVector')
                                # print(i)
                                # print(Objects1[n])
                                # print(displacement_previous[n])
                                ind_object = objs[0][n][i]
                                displacement_vector = self.displacement_previous[n][ind_object - 1]

                            # Extraction of tracking vector
                            vector_tracking = [pos[0], pos[1], math.sqrt(pos[0] * pos[0] + pos[1] * pos[1]),
                                               math.ceil(180 * pos[2] / math.pi),
                                               math.ceil(xc1), math.ceil(yc1), math.ceil(xc2 - xc1),
                                               math.ceil(yc2 - yc1), displacement_vector[0], displacement_vector[1]]

                            # Normalization of the vector
                            vector_tracking_norm = [vector_tracking[2] / self.norm_dist,
                                                    vector_tracking[3] / self.norm_ang,
                                                    2 * (vector_tracking[4] - (self.norm_x_img / 2)) / self.norm_x_img,
                                                    vector_tracking[5] / self.norm_y_img,
                                                    4 * vector_tracking[6] / self.norm_x_img,
                                                    4 * vector_tracking[7] / self.norm_y_img,
                                                    4 * vector_tracking[8] / self.norm_x_img,
                                                    4 * vector_tracking[9] / self.norm_y_img]
                            # Prediction of tracking-vector with SVM
                            vector_tracking_norm = np.array(vector_tracking_norm)
                            vector_tracking_norm = vector_tracking_norm.reshape(1, -1)

                            y_predict = int(self.model_svm.predict(vector_tracking_norm))
                            y_prob_positive = self.model_svm.predict_proba(vector_tracking_norm)[0, 1]
                            y_prob_negative = self.model_svm.predict_proba(vector_tracking_norm)[0, 0]

                            if y_predict == 1:
                                if y_prob_positive >= y_prob_negative:
                                    matrix_tracking_prob_act[i, j] = y_prob_positive

                    matrix_tracking_prob[n] = matrix_tracking_prob_act
                    # Print the results obtained
                    print('Clase:%s\n' % (self.labels[n]))
                    print("Matrix of Probability from Tracking")
                    print(matrix_tracking_prob[n])
                    print('-------------------------')
        # When finnish the checking of all classes the function returns the matrix created
        return matrix_tracking_prob

    ####################################################################################################################
    # FUNCTION FOR EXTRACT THE CORRESPONDENCE FROM MATRIX OF PROBABILITY
    ####################################################################################################################

    def extract_correspondence(self, cent1_objects, cent2_objects, Objs, pos, file_img1, file_img2,
                               matrix_track_prob, depth1_objects):

        # Loop for calculate extract the list with the correspondence
        for n in range(self.n_classes):
            # With this condition evaluate all objects except the detection of people
            if self.labels[n] != 'person':
                # Check if there are detected objects of the specific class in
                # both images.
                if cent1_objects[0][n] is not None and cent2_objects[0][n] is not None:
                    # ind_corresponding_xc1 indicate if the objects
                    # of the image1 have correspondence (1) or not(0).
                    # Initialization to 0.
                    ind_corresponding_xc1 = [0] * len(cent1_objects[0][n])

                    # ind_corresponding_xc2 indicate if the objects
                    # of the image2 have correspondence (1) or not (0).
                    # Initialization to 0.
                    ind_corresponding_xc2 = [0] * len(cent2_objects[0][n])

                    # Extraction of correspondences based on matrix of tracking probabilities.
                    # While there is more than one entry different to 0, we look for the
                    # maximum value of the matrix.
                    while np.any(matrix_track_prob[n]):
                        # Obtain the value max in the rows and columns from our probability matrix
                        row_max, col_max = np.where(matrix_track_prob[n] == np.max(matrix_track_prob[n]))
                        row_max = int(row_max)
                        col_max = int(col_max)
                        # Obtain the centroids
                        xc1 = cent1_objects[0][n][row_max]
                        yc1 = cent1_objects[1][n][row_max]
                        depth1 = depth1_objects[n][row_max]
                        xc2 = cent2_objects[0][n][col_max]
                        yc2 = cent2_objects[1][n][col_max]
                        # We check if none of the objects corresponding to row_max and col_max do not have
                        # correspondence. In that case, we extract a new correspondence.
                        if ind_corresponding_xc1[row_max] == 0 and ind_corresponding_xc2[col_max] == 0:

                            ind_corresponding_xc1[row_max] = 1
                            ind_corresponding_xc2[col_max] = 1
                            # Estimation pose of the objects
                            x_object, y_object = self.estimation_pose_object(xc1, depth1, pos[1])

                            print('Estimation object pose,%d,%d' % (row_max, col_max))
                            print(x_object, y_object)

                            # If there is the first correspondence of the object
                            if Objs[0][n][row_max] == 0:
                                self.ind_objects[n] += 1
                                Objs[0][n][row_max] = self.ind_objects[n]
                                Objs[1][n][col_max] = self.ind_objects[n]

                                # If there is the first correspondence within the class category, we create a dictionary
                                if self.objects_tracking[n] is None:
                                    self.objects_tracking[n] = [dict(Images=[file_img1, file_img2],
                                                                     x_coords=[xc1, xc2],
                                                                     y_coords=[yc1, yc2],
                                                                     pose_obj=[[x_object, y_object]])]
                                # If there is NOT the first correspondence within the class category, add a new
                                # dictionary
                                else:
                                    self.objects_tracking[n].extend([dict(Images=[file_img1, file_img2],
                                                                          x_coords=[xc1, xc2],
                                                                          y_coords=[yc1, yc2],
                                                                          pose_obj=[[x_object, y_object]])])

                                if self.displacement_previous[n] is None:
                                    self.displacement_previous[n] = []
                                    self.displacement_previous[n].append([math.ceil(xc2 - xc1), math.ceil(yc2 - yc1)])
                                else:
                                    self.displacement_previous[n].append([math.ceil(xc2 - xc1), math.ceil(yc2 - yc1)])
                                    # If there is NOT the first correspondence of the object, we add new data.
                            else:
                                ind_obj = Objs[0][n][row_max]
                                Objs[1][n][col_max] = ind_obj
                                # ind_obj goes from 1, 2,...The indexation will be ind_obj-1
                                # objects_tracking[n][ind_obj-1]['Images'].append(file_img1)
                                self.objects_tracking[n][ind_obj - 1]['Images'].append(file_img2)
                                # objects_tracking[n][ind_obj-1]['x_coords'].append(xc1)
                                self.objects_tracking[n][ind_obj - 1]['x_coords'].append(xc2)
                                # objects_tracking[n][ind_obj-1]['y_coords'].append(yc1)
                                self.objects_tracking[n][ind_obj - 1]['y_coords'].append(yc2)

                                self.objects_tracking[n][ind_obj - 1]['pose_obj'].append([x_object, y_object])

                                self.displacement_previous[n][ind_obj - 1] = [math.ceil(xc2 - xc1),
                                                                              math.ceil(yc2 - yc1)]
                        matrix_track_prob[n][row_max, col_max] = 0

                    print('------------------------------------')
                    print(Objs[0])
                    print('------------------------------------')
                    print(Objs[1])

    ####################################################################################################################
    # PRINT THE RESULTS FROM TRACKING
    ####################################################################################################################

    def results_tracking(self, sequence, y2):

        directory_results = './ResultsTracking/'
        # First: comprobate if the directory exist or not. In case to not exist, must be created
        if os.path.exists(directory_results):
            pass
        else:
            # Create the directories
            os.makedirs(directory_results, exist_ok=True)

        # Open the file
        fid = open(directory_results+'%s_Results.txt' % sequence, 'w')

        # Distance of Nodes
        fid.write("Distance nodes:%0.2f\n" % (y2 / 100))

        # Local vars for save the info of results
        estimation_pose_object = []
        dict_with_est_poses = {}

        for n in range(self.n_classes):
            print('----------------------------------------------------')
            print('Objects_tracking')
            print('Class:%s' % (self.labels[n]))

            if self.objects_tracking[n] is not None:
                for i in range(len(self.objects_tracking[n])):
                    print("Object:%d" % (i + 1))
                    d = self.objects_tracking[n][i]
                    print(d)
                    vector_distances_obj = []
                    # print('Estimation Pose')
                    for j in range(len(d['pose_obj'])):
                        # print(d['pose_obj'][j])
                        distance_obj = d['pose_obj'][j][1]
                        # Distances of more than 15 m are discarded.
                        if distance_obj is not None and distance_obj < 1600:
                            vector_distances_obj.append(d['pose_obj'][j][1])

                    # print('vector_distances_obj')
                    # print(vector_distances_obj)
                    if len(vector_distances_obj):
                        estimation_obj = statistics.median(vector_distances_obj)
                        # Generate a List with the poses of objects from same class
                        estimation_pose_object.append(estimation_obj/100)
                        print('-------------------------------------------')
                        print("estimation_obj:%0.4f (meters)" % (estimation_obj / 100))
                        print('-------------------------------------------')
                        fid.write("Class:%s, Distance (meters):%0.2f\n" % (self.labels[n], estimation_obj / 100))

                # Create the dictionary
                dict_with_est_poses[self.labels[n]] = estimation_pose_object
                # Reinit the var estimation_pose_object
                estimation_pose_object = []
        # Close the file when finnish the writing task
        fid.close()
        # Return the dictionary with the estimation pose from objects of sequence
        return dict_with_est_poses

    ####################################################################################################################
    # FUNCTION FOR VISUALIZE IN THE IMAGES THE PROCESS OF TRACKING OBJECTS
    ####################################################################################################################

    def visualize_tracking(self, sequence, first_img, last_img):

        directory_tracking_img = './ResultsTracking/ImgResultsTrack/'+sequence+'/'
        # First: comprobate if the directory exist or not. In case to not exist, must be created
        if os.path.exists(directory_tracking_img):
            pass
        else:
            # Create the directories
            os.makedirs(directory_tracking_img, exist_ok=True)

        # Path where is located the images
        path = './' + sequence + '/imgTestResults/ImgColor%d.png'

        for index in range(first_img + 1, last_img + 1):
            file_img = (path % index)
            img = cv2.imread(file_img)
            head_tail = os.path.split(file_img)
            filename_img_represent = head_tail[1]  # filename without path
            for n in range(self.n_classes):
                # print(objects_tracking[n])
                if self.objects_tracking[n] is not None:
                    for i in range(len(self.objects_tracking[n])):
                        d = self.objects_tracking[n][i]
                        # check if the tracked object includes the image
                        if file_img in d['Images']:
                            # If the tracked object includes the image, determine the position of the image
                            # in the tracked positions.

                            ind_img_object = d['Images'].index(file_img)
                            for j in range(ind_img_object):
                                x1 = d['x_coords'][j]
                                y1 = d['y_coords'][j]
                                x2 = d['x_coords'][j + 1]
                                y2 = d['y_coords'][j + 1]
                                line_thickness = 4
                                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=line_thickness)
                                img = cv2.circle(img, (x1, y1), radius=6, color=(255, 0, 0), thickness=-1)
                                img = cv2.circle(img, (x2, y2), radius=6, color=(255, 0, 0), thickness=-1)

                                # pos
                                # cad = str(j)
                                # pos = (x1, y1)
                                # img = cv2.putText(img, cad, pos, font,
                                # fontScale, color, thickness, cv2.LINE_AA)
                                # cad = str(j + 1)
                                # pos = (x2, y2)
                                # img = cv2.putText(img, cad, pos, font,
                                # fontScale, color, thickness, cv2.LINE_AA)

            # Write the image in the directory
            cv2.imwrite(directory_tracking_img + filename_img_represent, img)

    ####################################################################################################################
    # MAIN FUNCTION FOR EVALUATING A SEQUENCE OF IMAGES
    ####################################################################################################################

    def tracking_sequence(self, sequence, first_image=None, last_image=None):
        """
        This function evaluate the tracking objects for a sequence of images
        :param sequence: the path where is saved the sequence of images
        In case to evaluate a part of sequence this inputs delimit process of tracking
        :param first_image: first image of the sequence
        :param last_image: last image of the sequence
        :return:
        """
        # Path for detect de sequence of images, the files from encoder and file for depth
        path_imgs = './'+sequence+'/imgTestResults'
        path_enco = './'+sequence+'/Encoders'
        path_depth = './'+sequence+'/DataDepth'

        # Delimit the bounds for sequence
        if first_image is None and last_image is None:
            # In case that not detect any bounds for sequence, the method run the process for all images of sequence
            first_image = 0
            last_image = int(len(os.listdir(path_imgs)) / 3 - 1)

        # Loop to go through the sequence of images comparing pairs of images (the current one and the previous one)

        index = first_image

        while index < last_image:
            # File of Detections objects (First Image)
            file_img1 = path_imgs+'/ImgColor%d.png' % index              # File image with extension .png
            file_detection1 = path_imgs+'/ImgColor%d.txt' % index        # File of detections with extension .txt
            file_depth1 = path_depth+'/Img%d.txt' % index                 # File with the depth info with extension .txt

            # File of Detections objects (Second Image)
            file_img2 = path_imgs+'/ImgColor%d.png' % (index+1)           # File image with extension .png
            file_detection2 = path_imgs+'/ImgColor%d.txt' % (index+1)     # File of detections with extension .txt
            file_depth2 = path_depth + '/Img%d.txt' % (index+1)           # File with the depth info with extension .txt

            # Read the coordinates of the objects from YOLO
            xc1_objects, yc1_objects, depth1_objects = self.read_CoordYOLO(file_detection1, file_depth1)
            xc2_objects, yc2_objects, depth2_objects = self.read_CoordYOLO(file_detection2, file_depth2)

            # File of Encoders from wheelchair
            file_encoder1 = open((path_enco+'/Encoders%d.txt' % index), 'r')
            file_encoder2 = open((path_enco+'/Encoders%d.txt' % (index+1)), 'r')

            # Data from Encoder1 (Previous Image)
            encoder1 = file_encoder1.readline()
            encoder1 = encoder1.split('\n')[0]
            encoder1 = encoder1.split(',')
            # Obtain the coordinates from wheelchair (Previous State)
            X1 = float(encoder1[0].split('[')[1])
            Y1 = float(encoder1[1].split(' ')[1])
            T1 = float(encoder1[2].split(' ')[1].split(']')[0])

            # Data from Encoder2 (Actual Image)
            encoder2 = file_encoder2.readline()
            encoder2 = encoder2.split('\n')[0]
            encoder2 = encoder2.split(',')
            # Obtain the coordinates from wheelchair (Actual State)
            X2 = float(encoder2[0].split('[')[1])
            Y2 = float(encoder2[1].split(' ')[1])
            T2 = float(encoder2[2].split(' ')[1].split(']')[0])

            print('\n#############################################')
            print(file_img1)
            print(file_img2)
            print("inc_x:%0.3g,inc_y:%0.3g,inc_t:%0.3g" % (X2-X1, Y2-Y1, T2-T1))
            print('-------------------------')
            print("Coordinates of centroid's objects (x,y) from "+'ImgColor%d.png' % index)
            print(xc1_objects)
            print(yc1_objects)
            print('-------------------------')
            print("Coordinates of centroid's objects (x,y) from "+'ImgColor%d.png' % (index+1))
            print(xc2_objects)
            print(yc2_objects)
            print('-------------------------')
            ########################################################################################
            # Initialization of Objects1 and Objects2 (lists that store the indexes of objects)
            # For each class we set to 0 each one of the objects.
            if index is first_image:
                Objects2 = [None] * self.n_classes
                for i in range(self.n_classes):
                    if xc1_objects[i] is not None:
                        Objects2[i] = [0] * len(xc1_objects[i])

            Objects1 = Objects2

            Objects2 = [None] * self.n_classes
            for i in range(self.n_classes):
                if xc2_objects[i] is not None:
                    Objects2[i] = [0] * len(xc2_objects[i])
            #########################################################################################

            # List of variables for tracking matrix of probabilities
            pose = [X2-X1, Y2-Y1, T2-T1]
            pose_prev = [X1, Y1, T1]
            centroid1_objects = [xc1_objects, yc1_objects]
            centroid2_objects = [xc2_objects, yc2_objects]
            objects = [Objects1, Objects2]

            # Call the function that create the matrix of probabilities from tracking
            Matrix_Probability_Tracking = self.Tracking_Matrix_Probability(centroid1_objects,
                                                                           centroid2_objects,
                                                                           objects,
                                                                           pose)
            # Call the function that extract the correspondences using the matrix probability
            self.extract_correspondence(centroid1_objects,
                                        centroid2_objects,
                                        objects,
                                        pose_prev,
                                        file_img1,
                                        file_img2,
                                        Matrix_Probability_Tracking,
                                        depth1_objects)
            # Increment the index
            index = index + 1

        # These sentences go out from while loop
        # Process for obtain the results from tracking
        res_tracking_sequence = self.results_tracking(sequence, Y2)
        # Visualize images with tracked objects
        # Represent objects in the second image of each pair of consecutive frames
        self.visualize_tracking(sequence, first_image, last_image)

        return res_tracking_sequence

########################################################################################################################
# END CLASS DETECTION
########################################################################################################################


# ESTAS DOS SENTENCIAS IRIAN EN LA FUNCION MAIN
DET = Detection("classes.names", "YOLO_v1")
results = DET.tracking_sequence('Secuencia23')

print(results)

