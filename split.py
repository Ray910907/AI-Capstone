import os
import cv2
import numpy as np
from glob import glob
#Split images into 2 or 4 pieces
def segment_images(input_dir, output_pieces_dir, mode):
    """
    Split images into 2 or 4 pieces
    mode = 2: Left and right halves
    mode = 4: Four quadrants
    """
    #Set output directories,we use two/four different labels to represent the position of the photo pieces
    if mode == "sides":
        regions = ["left", "right"]
    elif mode == "corners":
        regions = ["top_left", "bottom_left", "top_right", "bottom_right"]
    else:
        raise ValueError("Mode must be 'corners' or 'sides'")
    
    for region in regions:
        os.makedirs(os.path.join(output_pieces_dir, region), exist_ok=True)
    
    image_paths = glob(os.path.join(input_dir, "*.jpg"))
    print("Image segmentation start")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path).split(".")[0]
        img = cv2.imread(img_path)
        
        #We get the height/width of the photo to cut the photo into several pieces
        h, w, _ = img.shape
        mid_h, mid_w = h // 2, w // 2
        
        if mode == "sides":
            left_piece = img[:, :mid_w]
            right_piece = img[:, mid_w:]
            
            cv2.imwrite(os.path.join(output_pieces_dir, "left", f"{filename}_left.jpg"), left_piece)
            cv2.imwrite(os.path.join(output_pieces_dir, "right", f"{filename}_right.jpg"), right_piece)
        
        elif mode == "corners":
            top_left = img[:mid_h, :mid_w]
            bottom_left = img[mid_h:, :mid_w]
            top_right = img[:mid_h, mid_w:]
            bottom_right = img[mid_h:, mid_w:]
            
            cv2.imwrite(os.path.join(output_pieces_dir, "top_left", f"{filename}_top_left.jpg"), top_left)
            cv2.imwrite(os.path.join(output_pieces_dir, "bottom_left", f"{filename}_bottom_left.jpg"), bottom_left)
            cv2.imwrite(os.path.join(output_pieces_dir, "top_right", f"{filename}_top_right.jpg"), top_right)
            cv2.imwrite(os.path.join(output_pieces_dir, "bottom_right", f"{filename}_bottom_right.jpg"), bottom_right)
        
        print(f"{filename} done!")
    
    print("Image segmentation done!")

#Input dir of the puzzle pieces then Enter the mode to represent the pieces of puzzle
input_dir = "dataset"
output_pieces_dir = "puzzle_pieces"
mode = input("Enter segmentation mode (sides: Left and right halves, corners: Four quadrants): ")

segment_images(input_dir, output_pieces_dir, mode)
