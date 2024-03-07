# TensorRT for Beginners: A Jupyter Notebook Walkthrough

Beginner-friendly tutorial for Tensor-RT using pyTorch on YOLO-V5.
Comparing deep learning compilation inference speed up and (dynamic) batching.

Video walkthrough and high-level explanation:

[![Youtube video link](https://img.youtube.com/vi/qAuRG0DBCNM/0.jpg)](https://youtu.be/qAuRG0DBCNM)


# Speeding Up Inference with TensorRT and YOLOv5

This Jupyter notebook demonstrates how to accelerate the inference process of YOLOv5 object detection model using NVIDIA's TensorRT. The notebook walks through the installation of necessary libraries, preparation of the COCO validation dataset, and execution of the model on a sample set of images. It also explores the process of converting the PyTorch model to a TensorRT-optimized model to achieve faster inference times.

## Prerequisites
- NVIDIA GPU with CUDA support
- PyTorch and Torch-TensorRT installed (will be installed in the notebook as well)

## Model Loading and Inference
The YOLOv5s model is loaded and preprocessed to run inference on a subset of the COCO validation dataset. The inference process is initially performed on a GPU to establish a baseline performance metric.

#### Accelerating Inference with TensorRT
The notebook further demonstrates how to convert the YOLOv5s model to a TensorRT-optimized model, reducing inference times significantly. A detailed comparison of inference times before and after optimization is provided.

#### Batching Inference
To improve efficiency, the notebook illustrates how to perform inference on batches of images. This approach leverages TensorRT's capabilities to further decrease the inference time per image.

#### Results
A comparison of the average inference times across different setups (baseline, TensorRT-optimized, and batched inference with TensorRT) is included, showcasing the performance gains achieved through optimization.
Finally, the notebook visualizes the speed-up achieved through TensorRT optimization and batching using a bar chart, providing a clear and comparative view of the performance improvements.
