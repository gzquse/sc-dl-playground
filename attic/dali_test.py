#!/usr/bin/env python3

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Define the DALI pipeline
class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id)
        # Reader to load files
        self.input = fn.readers.file(file_root=data_dir, random_shuffle=True)
        self.decode = fn.decoders.image(device="mixed", output_type=types.RGB)
        self.resize = fn.resize(resize_x=224, resize_y=224)
        self.normalize = fn.crop_mirror_normalize(
            device="gpu", dtype=types.FLOAT,
            output_layout=types.NCHW, mean=[0.5 * 255], std=[0.5 * 255]
        )

    def define_graph(self):
        # Define the graph of operations
        jpegs, labels = self.input()   # Load images and labels
        images = self.decode(jpegs)    # Decode the images
        resized = self.resize(images)  # Resize images to 224x224
        output = self.normalize(resized)  # Normalize the resized images
        return output, labels

# Pipeline parameters
batch_size = 32
data_dir = "./images"  # Replace with your image dataset path

# Initialize the pipeline
pipeline = SimplePipeline(batch_size=batch_size, num_threads=4, device_id=0, data_dir=data_dir)
pipeline.build()

# Wrap DALI pipeline into PyTorch DataLoader
data_loader = DALIGenericIterator([pipeline], ["data", "label"])

# Example training loop
for i, data in enumerate(data_loader):
    images = data[0]["data"]  # Images on GPU
    labels = data[0]["label"] # Labels
    print(f"Batch {i}: Images shape {images.shape}, Labels: {labels}")