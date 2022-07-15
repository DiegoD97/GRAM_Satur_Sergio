from numpy import e
import numpy as np
import copy
from map_annotation import Edge


class ResultsEvaluation(object):
    def __init__(self):
        # Attributes
        self.weights = []
        self.init_nodes = []
        self.dest_nodes = []
        self.possible_destinations = []
        self.historical_weights = []

    def empty_possible_destinations(self):
        if len(self.possible_destinations) == 0:
            return 1
        else:
            return 0


class EstimationWeights:
    ####################################################################################################################
    # BUILDER FOR CLASS
    ####################################################################################################################
    def __init__(self, file_ObjectClasses, topological_map=None):
        # Create the vars necessary for calculate the weights
        # LOAD THE OBJECT-CLASS LABELS OUR YOLO MODEL WAS TRAINED ON

        try:
            labelsPath = "Resources/" + file_ObjectClasses
            self.labels = open(labelsPath).read().strip().split("\n")
            self.n_classes = len(self.labels)  # Number of classes of objects
        except:
            print("File " + file_ObjectClasses + " is not found")
        # Constants gamma1 (objects) and gamma2 (nodes)
        self.gamma1 = 0.05
        self.gamma2 = 0.1

        # This dictionary is for save the data about the objects annotated that could be visualized with the
        # camera when the wheelchair navigate in the environment
        self.annotation_objects_vis = {
            'objects': {
                'window': {
                    'd_from_ref1': []
                },
                'door': {
                    'd_from_ref1': []
                },
                'elevator': {
                    'd_from_ref1': []
                },
                'fireext': {
                    'd_from_ref1': []
                },
                'plant': {
                    'd_from_ref1': []
                },
                'bench': {
                    'd_from_ref1': []
                },
                'firehose': {
                    'd_from_ref1': []
                },
                'lightbox': {
                    'd_from_ref1': []
                },
                'column': {
                    'd_from_ref1': []
                },
                'toilet': {
                    'd_from_ref1': []
                },
                'person': {
                    'd_from_ref1': []
                },
                'fallen': {
                    'd_from_ref1': []
                }
            }
        }

        # Create the Adjacency Matrix
        self.Matrix_Adyacencia = np.loadtxt('Matrix_Adyacencia.txt')
        # Create the Edges Matrix
        self.Matrix_Edges = np.loadtxt('Matrix_Edges.txt')
        shape_matrix = self.Matrix_Adyacencia.shape
        self.Matrix_Weights = np.zeros(shape_matrix)

        # Create the object for save the results from eval
        self.ResEval = ResultsEvaluation()

        # Load the annotated objects from topological map
        # self.annotated_objects = topological_map

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE WEIGHT (CORRESPONDENCE BETWEEN DETECTION AND ANNOTATION) FROM OBJECTS DETECTED
    ####################################################################################################################

    def weights_objects(self, edge_annotation, edge_detection, direction, dist_nodes_edge):
        # FIRST: in local var save the objects that the platform see. Depending on your movement
        # could not detect all the objects from annotation, so it is necessary filter the annotation
        # objects for the possible detection.
        # Loop for transit between the objects from the edge and know the objects that could be detected
        '''
        for classObject in edge_annotation['objects']:
            # Loop for transit between the different visualization objects
            for i in range(len(edge_annotation['objects'][classObject]['visualization'])):
                if (edge_annotation['objects'][classObject]['visualization'][i] == 0 or
                   edge_annotation['objects'][classObject]['visualization'][i] == direction):
                    # In case to visualize the object save in local variable the objects that could be
                    # detected
                    self.annotation_objects_vis['objects'][classObject]['d_from_ref1'].append(
                        edge_annotation['objects'][classObject]['d_from_ref1'][i]
                    )
        '''
        self.annotation_objects_vis = edge_annotation
        # Local var like security copy
        copy_annotation_objects_vis = copy.deepcopy(self.annotation_objects_vis)
        copy_edge_detection = copy.deepcopy(edge_detection)
        # Loop for calculate the weight of objects
        # Init the result product
        result_product = 1
        NC = 0
        for classObject_an in copy_annotation_objects_vis['objects']:
            for classObject_det in copy_edge_detection:
                # Case to detect the same object that in annotation
                if classObject_det == classObject_an:
                    # Check the lengths of list objects detected and annotated
                    if (len(copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1']) <=
                       len(copy_edge_detection[classObject_det]['d_from_ref1'])):
                        num_iter = len(copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1'])
                    else:
                        num_iter = len(copy_edge_detection[classObject_det]['d_from_ref1'])

                    for __ in range(num_iter):
                        # Study the correlation object
                        P1, P2, d = self.object_correlation(copy_annotation_objects_vis['objects'][classObject_an],
                                                            copy_edge_detection[classObject_det],
                                                            direction,
                                                            dist_nodes_edge)

                        result_product = result_product * (e ** (-self.gamma1 * d))
                        # Delete the value studied for next iteration
                        copy_annotation_objects_vis['objects'][classObject_an]['d_from_ref1'].pop(P2)
                        # Delete the value studied for next iteration
                        copy_edge_detection[classObject_det]['d_from_ref1'].pop(P1)

        # Finally, the objects of non-correspondence in the annotation and in the detection are counted.
        # NC for annotation
        for Object_an in copy_annotation_objects_vis['objects']:
            if len(copy_annotation_objects_vis['objects'][Object_an]['d_from_ref1']) != 0:

                NC += len(copy_annotation_objects_vis['objects'][Object_an]['d_from_ref1'])
        # NC for detection
        for Object_det in copy_edge_detection:
            if len(copy_edge_detection[Object_det]['d_from_ref1']) != 0:

                NC += len(copy_edge_detection[Object_det]['d_from_ref1'])

        # Calculate the result of products
        result_product = result_product * (0.8 ** NC)

        # Free the data from self.annotation_objects_vis for next iteration
        for ObjectVis in self.annotation_objects_vis['objects']:
            while (len(self.annotation_objects_vis['objects'][ObjectVis]['d_from_ref1'])) != 0:
                self.annotation_objects_vis['objects'][ObjectVis]['d_from_ref1'].pop(-1)

        return result_product

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE WEIGHT FROM NODES DETECTED
    ####################################################################################################################

    def weights_nodes(self, eval_node_init, eval_node_dest, node_annotation_init, node_annotation_dest, edge_annotated,
                      node_detect):
        # Check if the node detected and annotated at the init are the same
        if node_detect['Node_init'] == node_annotation_init['class']:
            # Now check if the node destiny are the same
            if node_detect['Node_dest'] == node_annotation_dest['class']:
                fj = 1
            else:
                fj = 0.35
        # In otherwise, where the init node not correspondence with the detection
        else:
            # In case that the final node correspondence between the detection and annotation
            if node_detect['Node_dest'] == node_annotation_dest['class']:
                fj = 0.35
            else:
                fj = 0.125
        # Determine the distance between nodes
        if ((edge_annotated['Ref1'] == eval_node_init and edge_annotated['Ref2'] == eval_node_dest) or
           (edge_annotated['Ref1'] == eval_node_dest and edge_annotated['Ref2'] == eval_node_init)):

            # Distance between nodes
            d_annotation = edge_annotated['dist']
            # Select the direction of movement
            if edge_annotated['Ref1'] == eval_node_init:
                direction_is = 1
            else:
                direction_is = -1

        distance = abs(node_detect['dist'] - d_annotation)

        result_product = fj * (e ** (-self.gamma2*distance))

        return result_product, direction_is

    ####################################################################################################################
    # FUNCTION FOR EVALUATE THE TOTAL WEIGHT CONSIDERING THE OBJECTS_WEIGHT AND NODES_WEIGHT
    ####################################################################################################################

    def evaluate_weight(self, Nodes_Annotated, Edges_Annotated, Nodes_Detect, Objects_Detect):

        max_weight = []
        Node_init = []
        Node_dest = []

        for i in Nodes_Annotated:
            for j in Nodes_Annotated:
                if (i != j) and (self.Matrix_Adyacencia[i-1][j-1] == 1):
                    edge = int(self.Matrix_Edges[i-1][j-1] - 1)
                    weight_nodes, direction_sel = self.weights_nodes(i, j,
                                                                     Nodes_Annotated[i],
                                                                     Nodes_Annotated[j],
                                                                     Edges_Annotated[edge],
                                                                     Nodes_Detect)

                    weight_objects = self.weights_objects(Edges_Annotated[edge], Objects_Detect,
                                                          direction_sel, Nodes_Detect['dist'])

                    total_weight = 0.5 * (weight_nodes + weight_objects)
                    self.Matrix_Weights[i-1][j-1] = total_weight

                    # if max_weight < total_weight:
                    #   max_weight = total_weight
                    #   Node1 = i
                    #   Node2 = j

        # Finished the process of calculate the matrix of weights now determine the posible values upper to 50 percent
        # to the max value

        for n in range(self.Matrix_Weights.shape[0]):
            for m in range(self.Matrix_Weights.shape[1]):
                if (n != m) and (self.Matrix_Weights[n][m] >= 0.5 * self.Matrix_Weights.max()):
                    max_weight.append(self.Matrix_Weights[n][m])
                    Node_init.append(n+1)
                    Node_dest.append(m+1)

        return max_weight, Node_init, Node_dest

    ####################################################################################################################
    # FUNCTION FOR EVAL THE LOCATION USING THE INFORMATION FROM 'n'-EDGES
    ####################################################################################################################

    def evaluate_location(self, Num_Edges_Eval, Nodes_Annotated, Edges_Annotated, Nodes_Detect, Objects_Detect):

        # FIRST: Evaluate the weight from each Edge for actual detection
        res = self.evaluate_weight(Nodes_Annotated, Edges_Annotated, Nodes_Detect, Objects_Detect)
        # Save the historical data from Evaluate in the object 'ResEval' for next iterations
        self.ResEval.weights.append(res[0])
        self.ResEval.init_nodes.append(res[1])
        self.ResEval.dest_nodes.append(res[2])

        # SECOND: calculate the possible location
        if not self.ResEval.empty_possible_destinations():
            # Auxiliar var initialize where save the actual weights
            actual_weight = []
            # In case to have more than one detection, we proceed to estimate the position using the information
            # of the possible destinations
            # 2.1) Calculate the weights only with the possible destinies obtained in the previous detection
            for n in range(len(self.ResEval.possible_destinations[-1])):
                if self.ResEval.possible_destinations[-1][n] in self.ResEval.dest_nodes[-1]:
                    # In case to coincide the possible destination of previous detection with destiny nodes of actual
                    # detection we proceed to calculate the weight
                    # 2.1.1) Obtain the previos weight
                    previous_weight = self.ResEval.historical_weights[-1][n]
                    # 2.1.2) Obtain the index for actual detection and know the weight to use
                    for m in range(len(self.ResEval.dest_nodes[-1])):
                        if self.ResEval.dest_nodes[-1][m] == self.ResEval.possible_destinations[-1][n]:
                            # Index of actual detection detected so finish this loop
                            index = m
                            break
                    # 2.1.3) Obtain the weight for current detection
                    current_weight = self.ResEval.weights[-1][index]
                    # 2.1.4) Calculate the weight
                    actual_weight.append(previous_weight * current_weight)

        else:
            actual_weight = self.ResEval.weights[-1]

        # THIRD: determine the posibles destinations using algebra of graphs used in the next iteration
        # Local var that is column matrix
        matrix_col_nodes = np.zeros((self.Matrix_Adyacencia.shape[0], 1))
        aux_list = []
        for i in range(len(self.ResEval.dest_nodes[-1])):
            # Actualize the value of column matrix
            node = self.ResEval.dest_nodes[-1][i]
            matrix_col_nodes[node-1][0] = 1
            # Calculate the possible destinations with our 'dest_node' from actual detection
            matrix_product = np.dot(self.Matrix_Adyacencia, matrix_col_nodes)

            # Loop to know the node/s destination for next iteration

            for j in range(matrix_product.shape[0]):
                if (matrix_product[j][0] == 1) and ((j+1) not in self.ResEval.dest_nodes[-1]):
                    aux_list.append(j+1)

            # Restart the matrix_col_nodes values for next iter of this loop
            matrix_col_nodes = np.zeros((self.Matrix_Adyacencia.shape[0], 1))

        # FOURTH: Update the value of the most important variables
        # Save the list of values for possible destinations
        self.ResEval.possible_destinations.append(aux_list)
        # Save the historial value of weights
        self.ResEval.historical_weights.append(actual_weight)

        # Delete the first value of list for create a FIFO of n-elements
        if len(self.ResEval.weights) == Num_Edges_Eval:
            self.ResEval.weights.pop(0)
            self.ResEval.init_nodes.pop(0)
            self.ResEval.dest_nodes.pop(0)

        if len(self.ResEval.possible_destinations) == Num_Edges_Eval:
            self.ResEval.possible_destinations.pop(0)
            self.ResEval.historical_weights.pop(0)

    ####################################################################################################################
    # FUNCTION FOR DETERMINE THE CORRELATION BETWEEN OBJECTS
    ####################################################################################################################
    @staticmethod
    def object_correlation(annotated_object, detected_object, direction, distance):
        # Local variables
        # The vars P1 and P2 contain the index from objects of same class with the minimum distance
        P1 = []
        P2 = []
        # Initialize the var error distance
        err = 1000

        # Loop for travel the dictionary from detection
        for index_obj_det in range(len(detected_object['d_from_ref1'])):
            # Loop for travel the dictionary from annotation
            for index_obj_an in range(len(annotated_object['d_from_ref1'])):
                # Observate the direction of platform
                if direction == 1:  # The platform goes from NodeRef1 to NodeRef2
                    dist_annotation = annotated_object['d_from_ref1'][index_obj_an]
                else:
                    dist_annotation = (distance - annotated_object['d_from_ref1'][index_obj_an])

                # Calculate the error distance between the distance from annotation and detection
                err_d = abs(detected_object['d_from_ref1'][index_obj_det] - dist_annotation)

                if err_d < err:
                    err = err_d
                    # Var for detection
                    P1 = index_obj_det
                    # Var for annotation
                    P2 = index_obj_an

        return P1, P2, err

########################################################################################################################
# END CLASS
########################################################################################################################

'''
edge_object_annotation_1 = {
    'Ref1': 1,

    'Ref2': 2,

    'dist': 18.04,

    'objects': {
        'window': {
                 'd_from_ref1': [19.5],
                 'visualization': [1]
                  },
        'door': {
                 'd_from_ref1': [1.0, 0.45, 1.63, 3.8, 5.06, 7.2, 8.4, 9.4, 10.7, 12.8, 12.8, 16.6, 16.6],
                 'visualization': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1]
                 },

                }
}

edge_object_annotation_2 = {
    'Ref1': 2,

    'Ref2': 3,

    'dist': 3.87,

    'objects': {
        'window': {
                 'd_from_ref1': [1.0, -1.87],
                 'visualization': [0, -1]
                 },
        'door': {
                 'd_from_ref1': [1.5, 4.7, 4.7, -0.72, -7.1],
                 'visualization': [0, 1, 1, -1, -1]
                 },
        'fireext': {
                        'd_from_ref1': [2.0],
                        'visualization': [0]
                        },

        'column': {
                   'd_from_ref1': [1.7, 4.65, -1.59, -5.05, -9.3],
                   'visualization': [0, 0, -1, -1, -1]
                   }

                }
}

edge_object_annotation_3 = {
    'Ref1': 2,

    'Ref2': 4,

    'dist': 7.36,

    'objects': {
        'window': {
                 'd_from_ref1': [1.13, 0.75, -1.98],
                 'visualization': [0, 0, -1]
                 },
        'door': {
                 'd_from_ref1': [0.84, 6.98, 11.89, -1.5, -4.7, -4.7],
                 'visualization': [0, 0, 0, -1, -1, -1]
                 },
        'plant': {
                        'd_from_ref1': [1.69],
                        'visualization': [1]
                        },

        'column': {
                   'd_from_ref1': [1.6, 5.09, 9.06, 12.8, -1.7],
                   'visualization': [0, 0, 0, 0, -1]
                   }

                }
}

edge_object_annotation_4 = {
    'Ref1': 4,

    'Ref2': 5,

    'dist': 3.89,

    'objects': {
        'window': {
                 'd_from_ref1': [4.77, 5.22, 4.96],
                 'visualization': [1, 1, 1]
                 },
        'door': {
                 'd_from_ref1': [3.0, 3.0, -0.55],
                 'visualization': [0, 0, -1]
                 },
        'plant': {
                 'd_from_ref1': [4.46],
                 'visualization': [1]
                },
                }
}

edge_object_annotation_5 = {
    'Ref1': 4,

    'Ref2': 6,

    'dist': 7.36,

    'objects': {
        'door': {
                 'd_from_ref1': [4.53],
                 'visualization': [0]
                 },

        'column': {
                   'd_from_ref1': [1.98, 5.28, 8.68],
                   'visualization': [0, 0, 1]
                   }

                }
}

edge_object_annotation_6 = {
    'Ref1': 6,

    'Ref2': 7,

    'dist': 18,

    'objects': {

        'door': {
                 'd_from_ref1': [5.17, 5.17, 7.4, 8.5, 9.7, 10.8, 13, 14, 16.5, 17.7, 19.1],
                 'visualization': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
                 },
        'fireext': {
                        'd_from_ref1': [6.5],
                        'visualization': [0]
                        },

        'lightbox': {
                   'd_from_ref1': [7.3],
                   'visualization': [0]
                   }

                }
}
'''

# DETECCION DE LA SECUENCIA 18

node_det_sec18 = {
    'Node_init': 'T',
    'Node_dest': 'E',
    'dist': 3.68
}

object_detection_sec18 = {

    'window': {
        'd_from_ref1': [4.66, 2.26, 4.51]
    },
    'door': {
             'd_from_ref1': [15.7]
             },
    'plant': {
        'd_from_ref1': [2.4, 2.71, 11.14]
    }
}

# DETECCION DE LA SECUENCIA 21

node_det_sec21 = {
    'Node_init': 'E',
    'Node_dest': 'T',
    'dist': 3.67
}

object_detection_sec21 = {

    'window': {
        'd_from_ref1': [1.6, 9.98]
    },
    'door': {
             'd_from_ref1': [8.75, 1.85]
             },
    'column': {
        'd_from_ref1': [1.96, 1.87, 5.29, 2.79, 2.15, 3.73, 4.3, 4.78, 4.78, 5.31]
    }
}

# DETECCION SECUENCIA 23

node_det_sec23 = {
    'Node_init': 'T',
    'Node_dest': 'E',
    'dist': 3.68
}

object_detection_sec23 = {

    'door': {
             'd_from_ref1': [3.54, 6.11, 1.76]
             },
    'column': {
        'd_from_ref1': [4.12]
    },
}

NODES = {
    1: {'class': 'E', 'pose': [200, 300]},
    2: {'class': 'T', 'pose': [250, 320]},
    3: {'class': 'E', 'pose': [200, 300]},
    4: {'class': 'T', 'pose': [200, 300]},
    5: {'class': 'E', 'pose': [200, 300]},
    6: {'class': 'T', 'pose': [200, 300]},
    7: {'class': 'E', 'pose': [200, 300]}
}

# Create the object from class Estimation
EW = EstimationWeights("classes.names")
# Create the object from class Edge where is saved the annotations objects
EDGE_AN = Edge()
EDGE_AN.read_nodes_txt('P_edge.txt')
EDGE_AN.read_objects_edge_txt()

# Create the edges with annotations
edge_obj_an_1 = {
    'Ref1': 1,
    'Ref2': 2,
    'dist': EDGE_AN.dist[2],
    'objects': EDGE_AN.edge_objects[0]
}

edge_obj_an_2 = {
    'Ref1': 2,
    'Ref2': 3,
    'dist': EDGE_AN.dist[0],
    'objects': EDGE_AN.edge_objects[1]
}

edge_obj_an_3 = {
    'Ref1': 2,
    'Ref2': 4,
    'dist': EDGE_AN.dist[1],
    'objects': EDGE_AN.edge_objects[2]
}

edge_obj_an_4 = {
    'Ref1': 4,
    'Ref2': 5,
    'dist': EDGE_AN.dist[4],
    'objects': EDGE_AN.edge_objects[3]
}

edge_obj_an_5 = {
    'Ref1': 4,
    'Ref2': 6,
    'dist': EDGE_AN.dist[1],
    'objects': EDGE_AN.edge_objects[4]
}

edge_obj_an_6 = {
    'Ref1': 6,
    'Ref2': 7,
    'dist': EDGE_AN.dist[2],
    'objects': EDGE_AN.edge_objects[5]
}

edge_obj_an_7 = {
    'Ref1': 2,
    'Ref2': 1,
    'dist': EDGE_AN.dist[2],
    'objects': EDGE_AN.edge_objects[6]
}

edge_obj_an_8 = {
    'Ref1': 3,
    'Ref2': 2,
    'dist': EDGE_AN.dist[0],
    'objects': EDGE_AN.edge_objects[7]
}

edge_obj_an_9 = {
    'Ref1': 4,
    'Ref2': 2,
    'dist': EDGE_AN.dist[1],
    'objects': EDGE_AN.edge_objects[8]
}

edge_obj_an_10 = {
    'Ref1': 5,
    'Ref2': 4,
    'dist': EDGE_AN.dist[4],
    'objects': EDGE_AN.edge_objects[9]
}

edge_obj_an_11 = {
    'Ref1': 6,
    'Ref2': 4,
    'dist': EDGE_AN.dist[1],
    'objects': EDGE_AN.edge_objects[10]
}

edge_obj_an_12 = {
    'Ref1': 7,
    'Ref2': 6,
    'dist': EDGE_AN.dist[2],
    'objects': EDGE_AN.edge_objects[11]
}

EDGES = [edge_obj_an_1, edge_obj_an_2, edge_obj_an_3,
         edge_obj_an_4, edge_obj_an_5, edge_obj_an_6,
         edge_obj_an_7, edge_obj_an_8, edge_obj_an_9,
         edge_obj_an_10, edge_obj_an_11, edge_obj_an_12]

node_det = node_det_sec23
object_det = object_detection_sec23
for __ in range(2):
    EW.evaluate_location(2, NODES, EDGES, node_det, object_det)
    object_det = object_detection_sec21
    node_det = node_det_sec21


# Peso, InitNod, DestNod = EW.evaluate_weight(NODES, EDGES, node_det_sec23, object_detection_sec23)
# print('El peso maximo es: '+str(Peso)+' y el viaje es de '+str(InitNod)+' ==> '+str(DestNod))








