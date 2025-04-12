import cv2
import numpy as np

# Create a white image for hand landmark visualization
def create_white_image():
    """
    Create a white image for hand landmark visualization
    
    Returns:
        A white image of size 400x400
    """
    white = np.ones((400, 400, 3), dtype=np.uint8) * 255
    cv2.imwrite('white.jpg', white)
    print("Created white.jpg for hand landmark visualization")

if __name__ == "__main__":
    create_white_image()
