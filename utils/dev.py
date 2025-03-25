import cv2
import numpy as np

def write_image(filename, img):
    img = (img.cpu().numpy() * 255).astype(np.uint8)
    path = r'./temp/' + filename + '.jpg'
    cv2.imwrite(path, img)
    print('saved', path)