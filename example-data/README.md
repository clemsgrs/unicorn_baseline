# Example data
This folder contains examples of the input and output interfaces used by the algorithm.

The algorithm takes as input a ``unicorn-task-description.json`` file, which contains detailed information about the specific task. This JSON file provides key details for processing input data, such as task descriptions, domain, modality, task type, and required inputs. Based on the information specified in the task description, the algorithm reads data from the corresponding folders as indicated by the fields ``input_socket_slug`` and ``relative_path``.

In the example provided in this folder:

- The algorithm will look for input data in ``images/prostate-tissue-biospy-wsi`` (whole-slide images) and ``images/tissue-mask`` (segmentation masks).
- The data includes a multi-resolution image and corresponding tissue masks (0 for background, 1 for tissue).
- The task is a pathology vision classification task.

The algorithm generates output data in two different formats depending on the task type:
   1. Slide-level features: ``image-neural-representation.json`` contains a single, one-dimensional vector of floats for each input scan/WSI. 
   2. Patch-level features: ``patch-neural-representation.json`` contains a list of patches, each with its extracted features and coordinates, and metadata such as spacing and patch size.
