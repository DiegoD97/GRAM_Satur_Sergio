import numpy as np
import os


class Edge(object):
    # Builder of class edge
    def __init__(self):
        # Path where is saved the data
        self.path = './Resources/'
        self.path_annotated = './Resources/Objetos_Anotados_Sur/'
        # Attributes of this class (Part of nodes)
        self.init_node = []
        self.dest_node = []
        self.dist = []
        # Attributes of this class (Part of annotated objects)
        self.class_object = {
            'window': {'d_from_ref1': []},
            'door': {'d_from_ref1': []},
            'elevator': {'d_from_ref1': []},
            'fireext': {'d_from_ref1': []},
            'plant': {'d_from_ref1': []},
            'bench': {'d_from_ref1': []},
            'firehose': {'d_from_ref1': []},
            'lightbox': {'d_from_ref1': []},
            'column': {'d_from_ref1': []}
        }
        self.edge_objects = []

    def read_nodes_txt(self, file_txt):
        # FIRST: open the file txt
        file = open(self.path+file_txt, 'r')

        # Loop for read line to line
        for line in file:
            data_split = line.split(' ')
            try:
                self.init_node.append(int(data_split[0]))
                self.dest_node.append(int(data_split[1]))
                self.dist.append(float(data_split[2].split('\n')[0]))
            except ValueError:
                pass

    def read_objects_edge_txt(self):
        # Reading the number of txt with the annotated objects
        num_files = len(os.listdir(self.path_annotated))
        # Read each file txt
        for i in range(num_files):
            # Load the txt
            file = np.loadtxt(self.path_annotated+'Edge_%d.txt' % (i+1))
            index_object = 0
            for Obj in self.class_object:
                obj_dist_ref1 = file[index_object]
                for j in range(len(obj_dist_ref1)):
                    if obj_dist_ref1[j] != 0:
                        self.class_object[Obj]['d_from_ref1'].append(obj_dist_ref1[j])
                # Increment the index for next iteration
                index_object = index_object + 1
            # Save the dictionary
            self.edge_objects.append(self.class_object)
            # Restart the dictionary for next iteration
            self.class_object = {
                'window': {'d_from_ref1': []},
                'door': {'d_from_ref1': []},
                'elevator': {'d_from_ref1': []},
                'fireext': {'d_from_ref1': []},
                'plant': {'d_from_ref1': []},
                'bench': {'d_from_ref1': []},
                'firehose': {'d_from_ref1': []},
                'lightbox': {'d_from_ref1': []},
                'column': {'d_from_ref1': []}
            }

########################################################################################################################
# END CLASS
########################################################################################################################
