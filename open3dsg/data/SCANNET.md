# ScanNet Dataset: Technical Overview

ScanNet is a comprehensive RGB-D video dataset comprising over 2.5 million views across 1,513 scans of indoor environments. Each scan directory is structured to include raw data, reconstructed meshes, and detailed annotations.

## Directory Structure

Each scan is stored in a directory named `scene_<spaceId>_<scanId>` (e.g., `scene0000_00`), containing the following files:

- **RGB-D Sensor Stream (`<scanId>.zip`) (`<scanId>.sens`)**: A compressed binary file containing color frames, depth frames, camera poses, and additional sensor data. (scans and scans_test, respectivately)

- **`scene<id>_<id>.txt`**: A text file containing metadata like axis alignment, RGB and depth frame dimensions, intrinsic camera parameters (e.g., focal lengths, principal points), color-to-depth transformation matrices, frame counts, and scene type for proper data interpretation and alignment.

- **High-Quality Reconstructed Mesh (`<scanId>_vh_clean.ply`)**: A binary PLY format mesh representing the scene's surface with the +Z axis oriented upright.

- **Cleaned and Decimated Mesh (`<scanId>_vh_clean_2.ply`)**: A simplified version of the reconstructed mesh, optimized for semantic annotation processes. 

- **Over-Segmentation Data (`<scanId>_vh_clean_2.0.010000.segs.json`)**: A JSON file detailing the over-segmentation of the annotation mesh, mapping each vertex to a segment index. 

- **Aggregated Semantic Annotations (`<scanId>.aggregation.json` or `<scanId>_vh_clean.aggregation.json`)**: JSON files providing instance-level semantic annotations, linking object instances to their corresponding segments and semantic labels. 

- **Visualization of Semantic Segmentation (`<scanId>_vh_clean_2.labels.ply`)**: A PLY file visualizing the aggregated semantic segmentation, with vertices colored according to NYU40 labels; includes a 'label' property denoting the NYU40 label ID. 

- **2D Projection Annotations**:
  - **Raw 2D Labels (`<scanId>_2d-label.zip`)**: 16-bit PNG images projecting aggregated annotation labels onto 2D frames, using ScanNet label IDs.
  - **Raw 2D Instances (`<scanId>_2d-instance.zip`)**: 8-bit PNG images projecting aggregated annotation instances onto 2D frames.
  - **Filtered 2D Labels (`<scanId>_2d-label-filt.zip`)**: Filtered versions of the raw 2D label projections.
  - **Filtered 2D Instances (`<scanId>_2d-instance-filt.zip`)**: Filtered versions of the raw 2D instance projections. :contentReference[oaicite:6]{index=6}

## Data Formats

- **Reconstructed Surface Mesh (`*.ply`)**: Binary PLY format meshes with vertices oriented such that the +Z axis is upright. 

- **RGB-D Sensor Stream (`*.sens`)**: Compressed binary files containing sequences of color and depth frames, along with camera poses and other sensor data. Parsing tools are available in the ScanNet C++ Toolkit. 

- **Surface Mesh Segmentation (`*.segs.json`)**: JSON files mapping each vertex to a segment index, facilitating over-segmentation analysis. 

- **Aggregated Semantic Annotations (`*.aggregation.json`)**: JSON files associating object instances with their constituent segments and semantic labels, providing comprehensive instance-level annotations. 
For detailed parsing examples and visualization tools, refer to the `BenchmarkScripts` directory in the ScanNet repository. 

