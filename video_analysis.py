import cv2
import numpy as np
import pandas as pd
from typing import List

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
    
    def forward(self, video_path: pd.DataFrame) -> pd.DataFrame:
        # List to store the results
        print(video_path)
        results = []
        
        # Iterate through the DataFrame of video paths
        for idx, row in video_path.iterrows():
            one_path = row['video_path']
            
            try:
                # Process each video and get the output path
                print("path is here",one_path,"\n")
                detected_objects = self.object_detector(one_path)
                # Add the result to the results list
                results.append({'output_video_path': detected_objects})
                
                # Log successful processing using the specific logger
                logger.info(f"Successfully processed {one_path}, output saved to {detected_objects}")
            
            except Exception as e:
                # Log any exceptions that occur during processing using the specific logger
                logger.error(f"Error processing {one_path}: {str(e)}")
        # Create a DataFrame from the results and return it
        return pd.DataFrame(results)
    
    def object_detector(self,input_path):
        output_folder = '/media/output/'
        output_path = os.path.join(output_folder, os.path.basename(input_path))

        # Construct command and call your script
        cmd = ['python3', 'process_clip.py', '--path', input_path, '--output', output_folder]
        process = subprocess.run(cmd, check=True)

        return output_path