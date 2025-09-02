import numpy as np
import cv2
import torch
import albumentations as A

from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import LineString


def predict_mask_grid(
    full_image,
    model,
    block_size,
    smoothing_type='overlap',
    overlap_px=32,
    gaussian_sigma=5
):
  """
  Receives a mask with road predictions based on satellite imagery

  :param full_image: satellite image (may be large)
  :param model: segmentation model
  :param block_size: side of the square block into which we divide full_image
  :param smoothing_type: ‘overlap’, i.e., dividing the image with overlap, averaging masks in places
      of overlap, or ‘gaussian’, when the division occurs without overlap,
      but then a Gaussian blur is added
  :param overlap_px: size of the overlap strip in pixels
  :param gaussian_sigma: Gaussian smoothing parameter
  :return: a mask of size (full_image_height, full_image_width) with values either 0 or 255
  """
  model.eval()

  height, width = full_image.shape[:2]
  mask_accum = np.zeros((height, width), dtype=np.float32)
  weight_mask = np.zeros((height, width), dtype=np.int64)

  step = block_size if smoothing_type == 'gaussian' else block_size - overlap_px

  for y in tqdm(range(0, height, step)):
    for x in tqdm(range(0, width, step), leave=False):
      real_h = min(block_size, height - y)
      real_w = min(block_size, width - x)

      tile = full_image[y:y+real_h, x:x+real_w, :]
      image = deepglobe_read_transform(image=tile)['image']
      image = deepglobe_val_transform(image=image)['image']

      with torch.no_grad():
        pred = model(image.to('cuda').unsqueeze(0))
        pred = pred.squeeze(0).squeeze(0).cpu().numpy()

      resize_back = A.Resize(height=real_h, width=real_w)
      pred = resize_back(image=pred)['image']

      mask_accum[y:y+real_h, x:x+real_w] += pred
      weight_mask[y:y+real_h, x:x+real_w] += 1

  weight_mask[weight_mask == 0] = 1

  if smoothing_type == 'overlap':
    final_mask = mask_accum / weight_mask
  else:
    final_mask = cv2.GaussianBlur(mask_accum, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)

  final_mask = np.where(final_mask >= 0.5, 255, 0).astype(np.int64)
  return final_mask


def cluster_mask_points(mask, eps=5, min_samples=1):
  """
  Clusters points from mask using the DBSCAN algorithm

  :param eps: DBSCAN parameter
  :param min_samples: DBSCAN parameter
  :return: clusters
  """
  coords = np.column_stack(np.where(mask > 0))
  if len(coords) == 0:
    return np.array([], dtype=int).reshape(0, 2)

  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
  labels = clustering.labels_

  clustered = defaultdict(list)
  for (y, x), label in tqdm(zip(coords, labels), leave=False):
    if label != -1:
      clustered[label].append((y, x))
  cluster_centers = [np.mean(pts, axis=0).astype(int) for pts in clustered.values()]

  return np.array(cluster_centers)


def plot_graph_overlay(background, G, figsize=(12, 12), title='', save_name='image'):
  """
  Draws the resulting graph G on top of the background

  :param background: image background
  :param G: graph
  :param figsize: graph size
  :param title: graph title
  :param save_name: file name for saving the graph
  """
  plt.figure(figsize=figsize)
  plt.imshow(background)
  plt.title(title)
  plt.axis('off')

  pos = nx.get_node_attributes(G, 'pos')

  for u, v, data in G.edges(data=True):
    line = data['geometry']
    x, y = line.xy
    plt.scatter(x, y, color=(0, 1, 1), s=5)

  x_nodes = [x for x, y in pos.values()]
  y_nodes = [y for x, y in pos.values()]
  plt.scatter(x_nodes, y_nodes, color=(0, 1, 0), s=13)
  plt.savefig(save_name)