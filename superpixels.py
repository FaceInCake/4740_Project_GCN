from pycocotools.coco import COCO
import numpy as np
import cv2
from skimage.segmentation import slic
from skimage.future import graph
# from skimage.future import show_rag
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json

# Path to the data
data_dir = './Data/'
annotation_file_training = data_dir + 'stuff_train2017.json'
annotation_file_val = data_dir + 'stuff_val2017.json'

image_dir_training = data_dir + 'train2017/'
image_dir_val = data_dir + 'val2017/'

segmentation_dir = data_dir + 'segmentations/'
val_segmentation_dir = data_dir + 'val_segmentations/'

os.makedirs(segmentation_dir, exist_ok=True)
#pixel_maps_dir_training = data_dir + 'stuff_train2017_pixelmaps/'


# Custom JSONEncoder that converts NumPy types to Python types
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(NumpyEncoder, self).default(obj)
        
        
def assign_labels_to_superpixels(annotations, segments, categories):
    # Initialize a dictionary to hold the category label for each superpixel
    superpixel_labels = {seg_id: np.zeros(len(categories)) for seg_id in np.unique(segments)}

    # Go through each annotation and assign category labels to the superpixels
    for ann in annotations:
        # Get the category id, and find the corresponding index for the category
        category_id = ann['category_id']
        category_index = next((index for (index, c) in enumerate(categories) if c['id'] == category_id), None)

        # Create a mask for the current annotation
        # If the annotation is in polygon format, you will need to convert it to mask
        ann_mask = coco.annToMask(ann) if ann['iscrowd'] == 0 else coco.decode(ann['segmentation'])

        # For each superpixel, check if it overlaps with the annotation mask
        for seg_id in superpixel_labels.keys():
            superpixel_mask = segments == seg_id
            overlap = ann_mask[superpixel_mask].sum()
            # If there is significant overlap, mark the superpixel with the category label
            if overlap > 0:  # This threshold can be adjusted
                superpixel_labels[seg_id][category_index] = 1

    return superpixel_labels

# Initialize COCO api for instance annotations
coco = COCO(annotation_file_val)

# Load the categories
categories = coco.loadCats(coco.getCatIds())
category_names = [cat['name'] for cat in categories]
print('COCO categories: \n{}\n'.format(' '.join(category_names)))

# Get image ids
image_ids = coco.getImgIds()

# # Load an image
# # for i, img_id in enumerate(image_ids[:3]):
# for img_id in image_ids:
#     image_info = coco.loadImgs(img_id)[0]
#     image = cv2.imread(image_dir_training + image_info['file_name'])
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     # Apply superpixel segmentation to the image using SLIC
#     segments = slic(image, n_segments=250, compactness=10, sigma=1, start_label=1)

#     # Create graph based on superpixel adjacency
#     g = graph.rag_mean_color(image, segments)

#     # Convert to networkx graph to apply GNNs
#     nx_graph = nx.Graph()
#     for region in g:
#         for neighbor in g[region]:
#             edge_weight = g[region][neighbor]['weight']  # Example property
#             nx_graph.add_edge(region, neighbor, weight=edge_weight)

#     # Assuming g is your region adjacency graph from skimage
#     nx_graph = nx.Graph()
#     for region in g:
#         coords = np.column_stack(np.where(segments == region))
#         centroid = np.mean(coords, axis=0)  # Calculate the centroid of the region
#         # Add each region as a node with a centroid attribute
#         nx_graph.add_node(region, centroid=centroid)

#         # Add edges with weights
#         for neighbor in g[region]:
#             edge_weight = g[region][neighbor]['weight']
#             nx_graph.add_edge(region, neighbor, weight=edge_weight)

#     # Create a plot with the original image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(image)

#     # For each node, find the centroid and draw
#     for region in nx_graph.nodes:
#         centroid = nx_graph.nodes[region]['centroid']  # Access the centroid attribute
#         ax.text(centroid[1], centroid[0], str(region), color='white', ha='center', va='center')

#     # Draw edges between regions
#     for (n1, n2) in nx_graph.edges:
#         p1 = nx_graph.nodes[n1]['centroid']
#         p2 = nx_graph.nodes[n2]['centroid']
#         ax.plot([p1[1], p2[1]], [p1[0], p2[0]], 'red')  # Draw a line between centroids

#     plt.show()





# Process each image
for img_id in image_ids:
    image_info = coco.loadImgs(img_id)[0]
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    
    image = cv2.imread(image_dir_val + image_info['file_name'])
    if image is None:
        continue  # If the image can't be loaded, skip to the next one
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply superpixel segmentation to the image using SLIC
    segments = slic(image, n_segments=250, compactness=10, sigma=1, start_label=1)

    # Create graph based on superpixel adjacency
    g = graph.rag_mean_color(image, segments)
    
    # Assign labels to each superpixel based on annotations
    superpixel_labels = assign_labels_to_superpixels(annotations, segments, categories)

    # Create a networkx graph to store RAG data
    nx_graph = nx.Graph()
    for region in g:
        # Node features could be the mean color within the superpixel
        # Convert numpy arrays to lists for JSON serialization
        node_features = np.mean(image[segments == region], axis=0)
        
        # Add each region as a node with the features attribute
        nx_graph.add_node(region, features=node_features, label=superpixel_labels[region])
        
        # Add edges with weights
        for neighbor, data in g.adj[region].items():
            edge_weight = data['weight']
            nx_graph.add_edge(region, neighbor, weight=edge_weight)

    # Save the graph to a JSON file
    graph_data = nx.node_link_data(nx_graph)  # Convert to dict for json serialization

    with open(os.path.join(val_segmentation_dir, f'image_{img_id}_graph.json'), 'w') as f:
        # Use a custom JSON encoder for numpy data types
        json.dump(graph_data, f, cls=NumpyEncoder)

