"""
YOLOv5 Inference Server

This module provides a command-line server for running YOLOv5 object detection.
It can process single images or entire folders and generates CSV files with detection results.

The server loads a YOLOv5 model once and keeps it in memory for multiple inference requests,
making it efficient for processing multiple images without reloading the model.

Key features:
- Interactive command-line interface
- Single image or batch folder processing
- Consolidated CSV output with detection results
- Support for GPU acceleration if available
- Proper resource cleanup when exiting
"""

import torch
import os
import yaml
import argparse
import signal
import traceback
import sys
import pandas as pd
import time
from pathlib import Path

class YOLOv5Inference:
    """
    YOLOv5Inference is a class for performing YOLOv5 object detection using PyTorch Hub.
    
    This class handles the model loading, inference, and result processing for YOLOv5 object detection.
    It provides methods for processing single images or entire directories and consolidates
    detection results into a CSV file.
    
    Attributes:
        model_path (str): Path to the YOLOv5 model weights (.pt file)
        labels_path (str): Path to the YAML file containing class labels
        enable_gpu (bool): Whether to use GPU for inference if available
        input_shape (list): Model input shape as [batch_size, channels, height, width]
        conf_thresh (float): Confidence threshold for detections (0-1)
        iou_thresh (float): IoU threshold for non-maximum suppression (0-1)
        detections_dir (str): Base directory for storing detection results
        base_dir (str): Project-specific directory for the current session
        csv_path (str): Path to the CSV file for storing detection results
        all_results (list): List to store pandas DataFrames of detection results
        class_labels (list): List of class names from the labels file
        model: The loaded YOLOv5 model
        device: PyTorch device (cuda or cpu)
    """
    
    def __init__(self, model_path, labels_path, enable_gpu=False, input_shape=[1,3,1280,1280], conf_thresh=0.25, iou_thresh=0.45, project_name=None):
        """
        Initialize the inference engine with model parameters.
        
        Args:
            model_path (str): Path to the YOLOv5 model weights (.pt file)
            labels_path (str): Path to the YAML file containing class labels
            enable_gpu (bool, optional): Whether to use GPU for inference if available. Defaults to False.
            input_shape (list, optional): Model input shape as [batch_size, channels, height, width]. Defaults to [1,3,1280,1280].
            conf_thresh (float, optional): Confidence threshold for detections (0-1). Defaults to 0.25.
            iou_thresh (float, optional): IoU threshold for non-maximum suppression (0-1). Defaults to 0.45.
            project_name (str, optional): Name for the output directory. Defaults to None (timestamp-based name).
        """
        # Store configuration parameters
        self.model_path = model_path
        self.labels_path = labels_path
        self.enable_gpu = enable_gpu
        self.input_shape = input_shape
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Create output directory structure
        self.detections_dir = "Detections"
        os.makedirs(self.detections_dir, exist_ok=True)
        
        # Set base directory name with timestamp if needed
        if project_name and not os.path.exists(os.path.join(self.detections_dir, project_name)):
            # Use provided project name if it doesn't exist already
            self.base_dir = os.path.join(self.detections_dir, project_name)     
        else:
            # Use timestamp-based name if no project name provided or it already exists
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.base_dir = os.path.join(self.detections_dir, f"results_{timestamp}")
        
        # Create directory structure
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Output file for CSV results
        self.csv_path = os.path.join(self.base_dir, "results.csv")
        
        # Store all detection results for consolidated CSV
        self.all_results = []
        
        # Parse labels from YAML file
        with open(self.labels_path, 'r') as f:
            labels = yaml.safe_load(f)
            self.class_labels = labels['names']
        
        # Load and configure YOLOv5 model from PyTorch Hub
        self.model = torch.hub.load("yolov5", "custom", path=self.model_path, source="local", _verbose=False)    
        self.device = torch.device("cuda" if self.enable_gpu and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.names = self.class_labels

        # Set detection parameters
        self.model.conf = self.conf_thresh  # Confidence threshold
        self.model.iou = self.iou_thresh    # IoU threshold for NMS
    
    def cleanup(self):
        """
        Properly clean up CUDA resources and save final consolidated results.
        
        This method should be called when shutting down the inference engine to ensure
        proper resource cleanup and finalization of results.
        """
        try:
            # Save all detection results to CSV
            self._save_consolidated_results()
            
            # Clear model from memory
            if hasattr(self, 'model') and self.model:
                self.model = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
    
    def _save_consolidated_results(self):
        """
        Save all detection results to a consolidated CSV file.
        
        This internal method concatenates all collected detection results and
        saves them to a single CSV file with columns ordered as specified.
        """
        if not self.all_results:
            return
        try:
            # Combine all detection results into a single DataFrame
            combined_df = pd.concat(self.all_results, ignore_index=True)
            
            # Reorder columns to desired format before saving
            ordered_columns = ['image', 'name', 'class', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
            available_columns = [col for col in ordered_columns if col in combined_df.columns]
            combined_df = combined_df[available_columns]
            
            # Save to CSV without index
            combined_df.to_csv(self.csv_path, index=False)
        except Exception as e:
            print(f"Error saving consolidated results: {str(e)}")
    
    def infer_single_image(self, image_path):
        """
        Run inference on a single image and save results.
        
        This method loads an image, runs YOLOv5 detection, saves the labeled image,
        and collects detection results for the CSV output.
        
        Args:
            image_path (str): Path to the image file to process
            
        Returns:
            object: The YOLOv5 results object or None if processing failed
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return None
            
            # Run YOLOv5 inference on the image
            results = self.model(image_path, size=[self.input_shape[2], self.input_shape[3]])
            
            # Save image with detection boxes drawn
            results.save(save_dir=self.base_dir, exist_ok=True)
            
            # Extract image basename and detection results
            img_name = os.path.basename(image_path)
            df = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
            
            # Add image filename to detection results
            df_with_img = df.copy()
            df_with_img["image"] = img_name
            
            # Store results if any detections were found
            if not df_with_img.empty:
                self.all_results.append(df_with_img)
            
            return results
            
        except Exception as e:
            traceback.print_exc()
            return None
    
    def infer_folder(self, folder_path):
        """
        Run inference on all images in a folder.
        
        This method processes all supported image files in the specified folder
        and saves consolidated results afterward.
        
        Args:
            folder_path (str): Path to the folder containing images to process
        """
        # Get list of image files from the folder
        image_paths = self.get_images_from_folder(folder_path)
        
        if not image_paths:
            return
        
        # Process each image in the folder
        for image_path in image_paths:
            try:
                self.infer_single_image(image_path)
            except Exception:
                traceback.print_exc()
        
        # Save all detection results to CSV
        self._save_consolidated_results()
    
    def get_images_from_folder(self, folder_path):
        """
        Get a list of image files from a folder.
        
        Args:
            folder_path (str): Path to the folder to scan for images
            
        Returns:
            list: List of full paths to image files
        """
        # Supported image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        
        try:
            # Scan directory for files with supported extensions
            for filename in os.listdir(folder_path):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    images.append(os.path.join(folder_path, filename))
        except Exception as e:
            print(f"Error listing directory {folder_path}: {str(e)}")
            
        return images

class YOLOv5Server:
    """
    A server that loads a YOLOv5 model once and processes multiple inference requests.
    
    This class provides a command-line interface for running YOLOv5 inference on
    images or folders. It initializes a YOLOv5Inference instance and handles user
    commands and graceful shutdown.
    
    Attributes:
        inferencer (YOLOv5Inference): The inference engine instance
        running (bool): Flag indicating whether the server is running
    """
    
    def __init__(self, model_path, labels_path, enable_gpu=False, input_shape=[1,3,1280,1280], conf_thresh=0.25, iou_thresh=0.45, project_name=None):
        """
        Initialize the YOLOv5 inference server.
        
        Args:
            model_path (str): Path to the YOLOv5 model weights (.pt file)
            labels_path (str): Path to the YAML file containing class labels
            enable_gpu (bool, optional): Whether to use GPU for inference if available. Defaults to False.
            input_shape (list, optional): Model input shape as [batch_size, channels, height, width]. Defaults to [1,3,1280,1280].
            conf_thresh (float, optional): Confidence threshold for detections (0-1). Defaults to 0.25.
            iou_thresh (float, optional): IoU threshold for non-maximum suppression (0-1). Defaults to 0.45.
            project_name (str, optional): Name for the output directory. Defaults to None (timestamp-based name).
        """
        # Create the YOLOv5 inference engine
        self.inferencer = YOLOv5Inference(
            model_path=model_path,
            labels_path=labels_path,
            enable_gpu=enable_gpu,
            input_shape=input_shape,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            project_name=project_name
        )
        
        # Server state
        self.running = False

    def start(self):
        """
        Start the server and listen for commands.
        
        This method initializes signal handlers, starts the command processing loop,
        and ensures proper cleanup on exit.
        """
        self.running = True
        # Register signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)
        try:
            # Start processing user commands
            self._process_commands()
        except KeyboardInterrupt:
            pass
        finally:
            # Ensure cleanup on exit
            self.running = False
            self.cleanup()

    def _handle_sigint(self, sig, frame):
        """
        Handle SIGINT (Ctrl+C) to shut down gracefully.
        
        Args:
            sig: Signal number
            frame: Current stack frame
        """
        self.running = False
        self.cleanup()

    def _process_commands(self):
        """
        Process user commands from the command-line interface.
        
        This method implements the main command loop of the server, parsing and
        executing user commands until the server is stopped.
        """
        while self.running:
            try:
                # Get command from user
                command = input("> ").strip().lower()
                
                if command.startswith("--image"):
                    # Process single image
                    args = command.split()
                    if len(args) < 2:
                        continue
                    
                    self._process_single_image(args[1])
                    
                elif command.startswith("--folder"):
                    # Process folder of images
                    args = command.split()
                    if len(args) < 2:
                        continue
                    
                    self._process_folder(args[1])

                elif command in ["quit", "exit", "q"]:
                    # Exit server
                    self.running = False

            except Exception:
                traceback.print_exc()

    def _process_single_image(self, image_path):
        """
        Process a single image with YOLOv5 detection.
        
        Args:
            image_path (str): Path to the image file to process
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                return
            
            # Run inference on the image
            self.inferencer.infer_single_image(image_path)

        except Exception:
            traceback.print_exc()

    def _process_folder(self, folder_path):
        """
        Process all images in a folder with YOLOv5 detection.
        
        Args:
            folder_path (str): Path to the folder containing images to process
        """
        try:
            # Validate folder path
            if not os.path.exists(folder_path):
                return
            
            # Run inference on all images in the folder
            self.inferencer.infer_folder(folder_path)

        except Exception:
            traceback.print_exc()

    def cleanup(self):
        """
        Properly clean up resources on server shutdown.
        
        This method ensures that the inference engine is properly cleaned up
        and resources are released when the server is shut down.
        """
        try:
            # Clean up the inference engine
            if hasattr(self, 'inferencer') and self.inferencer:
                self.inferencer.cleanup()
                self.inferencer = None
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
        self.running = False

def main():
    """
    Main entry point for the YOLOv5 PyTorch Inference Server.
    
    This function parses command-line arguments, initializes the server,
    and handles the main execution flow.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description='YOLOv5 PyTorch Inference Server')
    
    # Model and configuration arguments
    parser.add_argument('--model', required=True, type=str, help='Path to YOLOv5 weights file (.pt file)')
    parser.add_argument('--labels', required=True, type=str, help='Path to YAML file containing class labels')
    parser.add_argument('--enable_gpu', action='store_true', help='Enable GPU/CUDA inferences if device has NVIDIA GPU support (default: False)')
    parser.add_argument('--input_shape', type=str, default='1,3,1280,1280', help='Input shape as comma-separated values (default: 1,3,1280,1280)')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou_thresh', type=float, default=0.45, help='IOU threshold (default: 0.45)')
    parser.add_argument('--project_name', type=str, help='Output subdirectory name (default: timestamp-based name)')
    
    args = parser.parse_args()

    try:
        # Initialize the server with parsed arguments
        server = YOLOv5Server(
            model_path=args.model,
            labels_path=args.labels,
            enable_gpu=args.enable_gpu,
            input_shape=tuple(map(int, args.input_shape.split(','))),
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            project_name=args.project_name,
        )
        
        try:
            # Register cleanup function to run on exit
            import atexit
            atexit.register(lambda: server.cleanup() if 'server' in locals() and server else None)
            
            # Start the server
            server.start()
        finally:
            # Ensure cleanup happens even if there's an exception
            if 'server' in locals() and server:
                server.cleanup()
        return 0
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run the main function and use its return value as the exit code
    sys.exit(main())