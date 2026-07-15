import cv2
import numpy as np
# from other import thinning

from skimage.morphology import skeletonize

import matplotlib.pyplot as plt


def skeletonize_image(image):
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray=image
    # Threshold the image to obtain a binary image
    _, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Skeletonize the binary image
    skeleton = skeletonize(binary_image)

    return skeleton


def construct_graph(skeleton):
    import networkx as nx

    # Convert the skeleton image to a graph representation
    graph = nx.Graph()

    # Iterate over each pixel in the skeleton image
    rows, cols = skeleton.shape
    for y in range(rows):
        for x in range(cols):
            # If the pixel is part of the skeleton, add it as a node
            if skeleton[y, x]:
                graph.add_node((x, y))

                # Check neighboring pixels and add edges
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        # If the neighboring pixel is also part of the skeleton, add an edge
                        if 0 <= nx < cols and 0 <= ny < rows and skeleton[ny, nx]:
                            graph.add_edge((x, y), (nx, ny))

    return graph


def visualize_graph(graph):
    import networkx as nx
    # Visualize the graph
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


def skeleton_(val):
    # data1 = cv2.resize(val, (30, 30))
    conv_sktn_ = thinning.skeleton(val).astype('int32')
    nn = np.histogram(conv_sktn_, bins=50)[0]
    return nn
