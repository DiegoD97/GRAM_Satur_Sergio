import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import pi, sqrt, e
from rplidar import RPLidar
import time


# Class Lidar for proves
class Lidar (object):
    def __init__(self):
        self.ranges = []


class ResolutionError(Exception):
    pass


class LocalPlannerVFH:
    # Constructor from class VFH, here is initialized all of parameters
    def __init__(self):
        # LIDAR PARAMETERS ##############################################
        # Lidar object subscriber
        self.lidar = RPLidar('COM0')
        # Resolution Lidar (value of minimum angle detected in grades)
        self.resolutionLidar = 0.64748
        # Lidar cone Vision (in grades)
        self.LidarConeVision = 180      # Default value

        # Cone of collision
        self.cone_collision = self.LidarConeVision / 2  # Default value
        self.number_collisions = 0

        # PARAMETERS EQ m_ij = (c_ij)^2 * (a - b*d_ij) ###################
        self.a = 20
        self.b = 10

        # RELATIVE POSE ROBOT ############################################
        self.rel_pos_robot = [0, 0]

        # PARAMETERS FROM controllerVFH ##################################
        # Number of angular sectors in histogram
        self.NumAngularSectors = 180     # Default value

        # Limits for range readings in [m]
        self.DistanceLimits = [0.05, 2]  # Default value

        # Radios of vehicle in [m]
        self.RobotRadius = 0.2           # Default value

        # Safety distance around vehicle in [m]
        self.SafetyDistance = 0.4        # Default value

        # Minimum turning radius at current speed
        # self.MinTurningRadius = 0.1      # Default value

        # Cost function weight for target direction (constant)
        # Cost function weight for moving toward the target direction,
        # specified as a scalar. To follow a target direction, set this
        # weight to be higher than the sum of the CurrentDirectionWeight
        # and PreviousDirectionWeight properties. To ignore the target
        # direction cost, set this weight to zero.
        self.TargetDirectionWeight = 5    # Default value

        # Cost function weight for current direction (constant)
        # Cost function weight for moving the robot in the current
        # heading direction, specified as a scalar. Higher values
        # of this weight produce efficient paths. To ignore the current
        # direction cost, set this weight to zero.
        self.CurrentDirectionWeight = 2  # Default value

        # Cost function weight for previous direction (constant)
        # Cost function weight for moving in the previously selected
        # steering direction, specified as a scalar. Higher values of
        # this weight produces smoother paths. To ignore the previous
        # direction cost, set this weight to zero.
        self.PreviousDirectionWeight = 2  # Default value

        # Weight for penalize the effect of possibles collisions between
        # the robot with the possible objects detected. This value never
        # could be zero.
        self.PenalizeCostCollisions = 2   # Default value

        # Thresholds for binary histogram computation
        # Thresholds for binary histogram computation, specified as a
        # 2-element vector. The algorithm uses these thresholds to compute
        # the binary histogram from the polar obstacle density. Polar
        # obstacle density values higher than the upper threshold are
        # represented as occupied space (1) in the binary histogram.
        # Values smaller than the lower threshold are represented as free
        # space (0). Values that fall between the limits are set to the
        # values in the previous binary histogram, with the default being
        # free space (0).
        self.HistogramThresholds = [3, 10]  # Default value

        # PRINCIPAL VARS FROM CLASS THAT CONTAIN ALL INFO OF ALGORITHM
        self.lidar_filtering = []
        self.C_ij = []
        self.Beta_ij = []
        self.K = []
        self.K_collision = []
        self.Waves_Collision = []
        self.Sum_Waves_Distributions = []
        self.M_ij = []
        self.P_k = []
        self.G = []

        # Parameters that are used in the 'cost function' that evaluate the cost of the movement for avoid an object
        # the first var is used for give priority to the goal and the second var, determine the historic.
        self.targetDirection = 0.0      # Default value in radians
        self.previousDirection = 0.0    # Default value in radians

        # Variables for visualization results
        self.fig = plt.figure(figsize=(10, 5))
        self.ax1 = plt.subplot(1, 2, 1, projection='polar')
        self.ax2 = plt.subplot(1, 2, 2, projection='rectilinear')
        self.ax2.grid()

    ####################################################################################################################
    # FUNCTION FOR FILTERING THE LIDAR DATA
    ####################################################################################################################

    def filtering_data_lidar(self, read):
        """
        This function read the topic /scan and do a process of filtering data. This process,
        first extract the ray data for cone vision, then search the minimum measurement from
        each cone sector and finally return a list with this data
        :param read: parameter from topic /scan
        :return: data filtering from lidar
        """
        # RESTART THE VALUE WHERE IS SAVED THE LIDAR DATA FILTERED
        self.lidar_filtering = []
        try:
            # First calculate the number of measures by sector
            measures_by_sector = int(self.LidarConeVision / (self.NumAngularSectors * self.resolutionLidar))

            if measures_by_sector == 0:
                raise ResolutionError

        except ResolutionError:
            print("ERROR -> Resolution VFH under resolution from LIDAR")
            # Assign the minimum value to measures by sector
            measures_by_sector = 1
            pass

        # Middle value from cone vision
        middle_cone = int(self.LidarConeVision / (2 * self.resolutionLidar))

        # Read the useful data for cone between the 0 to -90º
        middle_right = read.ranges[0:middle_cone]
        # Read the useful data for cone between the 0 to 90º
        middle_left = read.ranges[(len(read.ranges) - middle_cone):len(read.ranges)]
        # Create a result list concatenate it
        results_lidar = middle_left.tolist() + middle_right.tolist()
        # After adjust the value of measures_by_sector, with the length of array results_lidar,
        # is necessary adjust the value of number of angular sectors
        self.NumAngularSectors = int(len(results_lidar) / measures_by_sector)

        # Go through each sector
        for i in range(int(len(results_lidar) / measures_by_sector)):
            # First measure for sector
            init = i * measures_by_sector
            # Final measure for sector
            final = (i + 1) * measures_by_sector
            # Create an auxiliar list for then be sorted from minimum to maxim value
            aux_list = results_lidar[init:final]
            res = sorted(aux_list)
            # Save the minimum measure from cone sector
            self.lidar_filtering.append(res[0])

        if len(self.lidar_filtering) > self.NumAngularSectors:
            self.lidar_filtering = self.lidar_filtering[0:(len(self.lidar_filtering) - 1)]

    ####################################################################################################################
    # FUNCTION THAT INTEGRATES ALL STEPS FOR DEVELOP THE VFH+ CONTROLLER
    ####################################################################################################################

    def controllerVFH(self, show_res=False):
        # Use the block try-except for capture the errors and avoid the lock of program
        try:
            # RESTART THE VALUES FOR EACH CYCLE OF SCAN (IMPORTANT)
            self.Beta_ij = []
            self.C_ij = []
            self.K = []
            self.K_collision = []
            self.Waves_Collision = []
            self.Sum_Waves_Distributions = []
            self.M_ij = []
            self.P_k = []
            self.G = []

            # LOOP FOR CALCULATE THE PARAMETERS Beta_ij, C_ij, K AND M_ij
            for i in range(len(self.lidar_filtering)):
                # FIRST STEP: calculate the values Beta_ij of algorithm
                self.Beta_ij.append(self.beta_ij_value(i))
                # SECOND STEP: calculate the values C_ij of algorithm
                self.C_ij.append(self.c_ij_value(self.lidar_filtering[i]))
                # THIRD STEP: calculate the factor K (must be an integer)
                self.K.append(int(self.Beta_ij[-1] / (self.LidarConeVision / self.NumAngularSectors)))
                # FOURTH STEP: calculate the factor K_collision
                self.create_K_collision(self.Beta_ij[-1], self.K[-1])
                # FIFTH STEP: calculate the parameter M_ij of algorithm
                self.M_ij.append(self.m_ij_value(self.C_ij[-1], self.lidar_filtering[i]))

            ######################################
            # EVALUATE FOR VALUE ORIENTATION 0º IN CASE NUMBER PAR OF SECTORS
            if self.NumAngularSectors % 2 == 0:
                # Value Beta_ij
                self.Beta_ij.append(0)
                # Sorted the list
                self.Beta_ij = sorted(self.Beta_ij)
                # Value C_ij using the contiguous sectors to 0º
                C_ij_prev_cero = self.C_ij[0:int(self.NumAngularSectors / 2)]
                C_ij_post_cero = self.C_ij[int(self.NumAngularSectors / 2):len(self.C_ij)]
                if C_ij_prev_cero[-1] >= C_ij_post_cero[0]:
                    C_ij_cero = [C_ij_prev_cero[-1]]
                else:
                    C_ij_cero = [C_ij_post_cero[0]]
                self.C_ij = C_ij_prev_cero + C_ij_cero + C_ij_post_cero
                # Value K
                self.K.append(0)
                # Sorted the list K
                self.K = sorted(self.K)
                # Value of K_collision
                self.K_collision.append(0)
                self.K_collision = sorted(self.K_collision)
                # Calculate the value of M_ij
                M_ij_prev_cero = self.M_ij[0:int(self.NumAngularSectors / 2)]
                M_ij_post_cero = self.M_ij[int(self.NumAngularSectors / 2):len(self.M_ij)]
                if M_ij_prev_cero[-1] >= M_ij_post_cero[0]:
                    M_ij_cero = [M_ij_prev_cero[-1]]
                else:
                    M_ij_cero = [M_ij_post_cero[0]]

                self.M_ij = M_ij_prev_cero + M_ij_cero + M_ij_post_cero
            ###################################

            # LOOP FOR CALCULATE THE PROBABILISTIC OCCUPANCY OF THE FRONTAL ROBOT AND DETERMINE THE POSSIBLE
            # COLLISIONS INCLUDING THE CONCEPT OF 'WAVE COLLISION'
            for j in range(len(self.M_ij)):
                # SIXTH STEP: calculate the probabilistic occupancy P_k
                self.P_k.append(self.p_k_value(self.M_ij, self.K[j]))
                # SEVENTH STEP: determine the security for move the robot using the cone collision
                self.determine_wave_collisions(self.P_k[-1], self.K[j], index=j)

            # EIGHT STEP: plus the values of lists where is saved the value os waves collisions
            self.sum_n_lists()

            # LOOP FOR CALCULATE THE VALUE OF FUNCTION OF COST
            for n in range(len(self.P_k)):
                # NINETH STEP: calculate the cost function G
                res = self.cost_function(self.P_k[n], self.Beta_ij[n], self.Sum_Waves_Distributions[n])
                # In case that the 'res' is different None, save the value
                if res is not None:
                    self.G.append(res)
                else:
                    pass

            # FINAL STEP: determine the minimum cost for avoid the object
            minimumCost = min(self.G)
            # Print the result is optional
            if show_res:
                print("The minimum cost is: "+str(minimumCost[0])+". Turning with: "+str(minimumCost[1])+'º')
                self.showPolarOccupancy(minimumCost[1])

        except ValueError:
            print("An error has occurred")

    ####################################################################################################################
    # FUNCTION FOR CALCULATE THE VALUE OF Cij
    ####################################################################################################################

    def c_ij_value(self, rays_filter):
        """
        This function give the values for knowing the state from each cone sector. There are 4 posible
        states.
        C[i] = 3 -> Represent a forbidden zone
        C[i] = 2 -> Is a possible free zone but is not secure
        C[i] = 1 -> Is a possible free zone secure
        C[i] = 0 -> Is a free space
        :param rays_filter: data from self.lidar_filtering[index]
        :return: is a value for each sector
        """
        # Check the distance of rays for determine the value
        if rays_filter <= (self.RobotRadius + self.DistanceLimits[0]):
            # Forbidden zone
            return 3
        elif ((rays_filter > (self.RobotRadius + self.DistanceLimits[0])) and
              (rays_filter <= (self.SafetyDistance + self.RobotRadius))):
            # Possible free space but is not secure
            return 2
        elif ((rays_filter > (self.SafetyDistance + self.RobotRadius)) and
              (rays_filter < (self.DistanceLimits[1] + self.RobotRadius))):
            # Possible free space secure
            return 1
        else:
            # Free space secure
            return 0

    ####################################################################################################################
    # FUNCTION THAT DETERMINE THE PARAMETER BETA
    ####################################################################################################################

    def beta_ij_value(self, index_lidar_filtering):
        """
        This function calculates the bisector of each sector
        :param index_lidar_filtering: is the index from loop 'controllerVFH'
        :return: the parameter beta is the angle between relative pose robot and cone sector
        """
        angle_sector = ((self.LidarConeVision / (2 * self.NumAngularSectors)) +
                        index_lidar_filtering * (self.LidarConeVision / self.NumAngularSectors) -
                        (self.LidarConeVision / 2))

        return angle_sector

    ####################################################################################################################
    # FUNCTION THAT DETERMINE THE PARAMETER M_ij
    ####################################################################################################################

    def m_ij_value(self, c_ij, rays_filter):
        """
        This function calculate the parameter m_ij from algorithm
        :param c_ij: is the value of occupancy
        :param rays_filter: is the value of distance of each sector
        :return:
        """
        return (c_ij**2) * abs(self.a - self.b * rays_filter)

    ####################################################################################################################
    # FUNCTION THAT DETERMINE THE PROBABILISTIC OCCUPANCY FOR FRONTAL ROBOT
    ####################################################################################################################

    def p_k_value(self, m_ij, K):
        """
        This function calculate the probabilistic occupancy of algorithm
        :param m_ij: is the list M_ij
        :param K: is the index sector
        :return:
        """
        suma = 0.0
        # Loop to go through all the values of the arrays m_ij
        for j in range(len(m_ij)):
            if K == int(self.Beta_ij[j] / (self.LidarConeVision / self.NumAngularSectors)):
                suma += m_ij[j]
        return suma
    ####################################################################################################################
    # FUNCTION FOR CREATE THE LIST K_collision
    ####################################################################################################################

    def create_K_collision(self, beta_ij, k):
        if (beta_ij >= -(self.cone_collision / 2)) and (beta_ij <= (self.cone_collision / 2)):
            # Save the value of K in the list K_collision
            self.K_collision.append(k)

    ####################################################################################################################
    # FUNCTION FOR DETERMINE THE WAVE DISTRIBUTION COLLISION
    ####################################################################################################################

    def determine_wave_collisions(self, p_k, k, index):
        # Determine the number cones collisions that interference with cones with high probability of
        # occupancy
        if (p_k >= self.HistogramThresholds[1]) and (k in self.K_collision):
            # Value of list distribution of wave collision
            self.Waves_Collision.append(self.distribution_wave(self.Beta_ij[index], self.C_ij[index]))

    ####################################################################################################################
    # FUNCTION FOR CALCULATE THE DISTRIBUTION FOR WAVE COLLISION
    ####################################################################################################################

    def distribution_wave(self, angle_value, ampl_normal):
        # Auxiliar list
        res_list = []
        angle_per_sector = self.LidarConeVision / self.NumAngularSectors
        # Loop for calculate the values of distribution wave
        for i in range(len(self.Beta_ij)):
            try:
                calc_res = ampl_normal/(sqrt(e ** ((self.Beta_ij[i]-angle_value)/(2 * angle_per_sector))**2))
            except OverflowError:
                calc_res = 0
                pass

            res_list.append(calc_res)

        return res_list

    ####################################################################################################################
    # FUNCTION FOR PLUS THE VALUES OF 'N' LISTS
    ####################################################################################################################

    def sum_n_lists(self):
        self.Sum_Waves_Distributions = [0.0] * len(self.Beta_ij)

        for i in range(len(self.Waves_Collision)):
            for j in range(len(self.Waves_Collision[0])):
                self.Sum_Waves_Distributions[j] += self.Waves_Collision[i][j]

    ####################################################################################################################
    # FUNCTION THAT DETERMINE THE COST OF MOVEMENT FOR AVOID AN OBJECT
    ####################################################################################################################

    def cost_function(self, p_k, beta_ij, wave_collisions_value):
        """
        Function for calculate the cost of movement for avoid the object
        :param wave_collisions_value: this parameter represents the value from list of wave
        collisions distribution
        :param p_k: value of probabilistic occupancy, if the value is over the threshold the space is
        considered occupied
        :param beta_ij: value of sector to turn the robot
        :return:
        """
        # Comprobate the thresholds
        if p_k <= self.HistogramThresholds[1]:
            # First obtain the angle to evaluate the cost when change the wheel orientation
            WheelOrient = (beta_ij * 3.141592 / 180)  # Covert to radians
            # Then, calculate the cost
            fcnCost = (self.TargetDirectionWeight * self.targetDirection +
                       self.CurrentDirectionWeight * abs(WheelOrient - 0) +
                       self.PreviousDirectionWeight * abs(self.previousDirection) +
                       self.PenalizeCostCollisions * wave_collisions_value)

            if p_k < self.HistogramThresholds[0]:
                # Free zone with high security
                return [fcnCost, beta_ij]
            else:
                # Posible free zone, the p_k is between minimum and maxim threshold, so
                # penalize the function cost with a constant
                return [1.05*fcnCost, beta_ij]
        else:
            # The value is over the threshold so return None
            return None

    ####################################################################################################################
    # FUNCTION FOR SHOWING THE POLAR SECTOR OF OCCUPANCY OBJECTS
    ####################################################################################################################

    def showPolarOccupancy(self, turning_ang):
        # Compute pie slices
        # FIRST STEP: covert degrees to radians
        theta = []
        for i in range(len(self.Beta_ij)):
            theta.append(self.Beta_ij[i] * pi / 180)
        # SECOND STEP: determine the radius for polar diagram
        radii = self.P_k
        # THIRD STEP: determine the with of each sector in radians
        width = (self.LidarConeVision / self.NumAngularSectors) * pi / 180
        # FOURTH STEP: plot and show the graphic
        self.ax1.bar(theta, radii, width=width)
        self.ax1.plot([-self.cone_collision*pi/(2*180), -self.cone_collision*pi/(2*180)],
                      [0, max(radii)], color="r", linewidth=2)
        self.ax1.plot([self.cone_collision*pi/(2*180), self.cone_collision*pi/(2*180)],
                      [0, max(radii)], color="r", linewidth=2)
        self.ax1.plot([0, turning_ang*pi/180], [0, 0.5*max(radii)], color="g", linewidth=2)

        self.ax2.plot(self.Beta_ij, self.Sum_Waves_Distributions)
        self.ax2.axvline(x=turning_ang, ymin=0.0, ymax=0.5, color='g')

        self.fig.canvas.draw()
        image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow('Occupancy Probability', image)
        cv2.waitKey(10)

########################################################################################################################
# FINAL CLASS
########################################################################################################################


# Create the objects
rplidar = Lidar()
VFH = LocalPlannerVFH()

VFH.NumAngularSectors = 300
VFH.RobotRadius = 0.2
VFH.DistanceLimits = [0.05, 2.0]
VFH.LidarConeVision = 175
VFH.cone_collision = 130
VFH.HistogramThresholds = [3, 30]
VFH.PenalizeCostCollisions = 10

while True:
    rplidar.ranges = np.random.random(556)
    for p in range(len(rplidar.ranges)):
        rplidar.ranges[p] = rplidar.ranges[p] * 8.0

    VFH.filtering_data_lidar(rplidar)
    VFH.controllerVFH(show_res=True)







