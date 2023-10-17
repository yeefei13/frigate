import cv2
import numpy as np
import pandas as pd
from typing import List
import json
import shutil

from evadb.functions.abstract.abstract_function import AbstractClassifierFunction
from evadb.functions.gpu_compatible import GPUCompatible
from evadb.utils.logging_manager import logger
import subprocess
import os


class ObjectDetector(AbstractClassifierFunction, GPUCompatible):
    def setup(self):
        pass
    def load_labels(self, filepath):
        labels = []
        with open(filepath, 'r') as file:
            for line in file:
                # split the line by whitespace and take all elements after the first one
                label_parts = line.strip().split()[1:]  
                # join them together and append to labels list
                labels.append(" ".join(label_parts))  
        return labels
    @property
    def name(self) -> str:
        return "ObjectDetector"
    
    @property
    def labels(self) -> List[str]:
        filepath = 'labelmap.txt'
        return self.load_labels(filepath)
    
    @property
    def to_device(self):
        pass
    
    def forward(self, video_paths: pd.DataFrame) -> pd.DataFrame:
        # List to store the results
        results = []

        # Create a list to store the video paths
        input_paths = video_paths['video_path'].tolist()
        
        try:
            # Process all video clips and get the output paths
            detected_objects = self.object_detector(input_paths)
            
            # Log successful processing using the specific logger
            logger.info(f"Successfully processed all video clips, output saved to {detected_objects}")
            
            # Create a DataFrame with input and output paths
            for input_path, output_path in zip(input_paths, detected_objects):
                results.append({'output_video_path': output_path})
        
        except Exception as e:
            # Log any exceptions that occur during processing using the specific logger
            logger.error(f"Error processing video clips: {str(e)}")

        # Create a DataFrame from the results and return it
        return pd.DataFrame(results)

    def object_detector(self, input_paths):
        output_folder = '/media/output/'
        output_paths = []

        # Construct the command to process all input paths at once
        cmd = ['python3', 'process_clip.py', '--path'] + input_paths + ['--output', output_folder]

        # Capture the output of the subprocess
        try:
            completed_process = subprocess.run(cmd, check=True, capture_output=True, text=True)

            # Check if the subprocess ran successfully
            if completed_process.returncode == 0:
                # Define the path to the generated CSV file
                generated_csv_path = os.path.join(output_folder, 'detection_results.csv')

                # Copy the generated CSV file to the output folder
                shutil.copy('path_of_generated_csv_file.csv', generated_csv_path)

                # Read the CSV file to extract the "file" column
                df = pd.read_csv(generated_csv_path)
                
                # Append output paths for all input paths based on the "file" column
                output_paths.extend(df['file'].tolist())
        except subprocess.CalledProcessError as e:
            # Handle any errors that occurred during subprocess execution
            print(f"Error running the subprocess: {e}")

        return output_paths

