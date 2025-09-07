from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import cv2

def make_square(image):
    height, width = image.shape
    if height > width:
        padding = int((height - width)/2)
        new_image = np.zeros((height, height), np.uint8)
        new_image[0:height, padding : padding + width] = image
    else:
        padding = int((width - height)/2)
        new_image = np.zeros((width, width), np.uint8)
        new_image[padding : padding + height, 0 : width] = image
    return new_image

def cloth_segment(input_image):
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    image = Image.open(input_image)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    # output = 'cloth/3.jpg'
    pred_seg = np.array(pred_seg * 10).astype('uint8')
    cv2.imwrite("cloth/mask.jpg", pred_seg)
    mask1 = np.zeros(pred_seg.shape).astype('uint8')
    mask2 = np.zeros(pred_seg.shape).astype('uint8')
    # print(np.unique(pred_seg))
    mask1[np.where(pred_seg == 40)] = 255
    mask2[np.where(pred_seg == 70)] = 255
    mask2[np.where(pred_seg == 60)] = 255
    mask2[np.where(pred_seg == 50)] = 255
    # mask1 = make_square(mask1)
    # mask2 = make_square(mask2)
    # tile1 = cv2.bitwise_and(tile1, tile1, mask = mask)
    input_img = cv2.imread(input_image)
    size = input_img.shape
    image_bg = np.ones((size[0], size[1], 3), np.uint8) * 255
    
    image_up = cv2.bitwise_and(input_img, input_img, mask = mask1)
    image_bg_up = cv2.bitwise_and(image_bg, image_bg, mask = cv2.bitwise_not(mask1))
    image_up = image_up + image_bg_up
    # cv2.imshow("mask_up", mask1)
    
    x, y, w, h = get_rectangle(mask1, image_up, 1.1)
    image_up_1 = cropped_image(x, y, w, h, image_up)
    x, y, w, h = get_rectangle(mask1, image_up, 1.3)
    image_up_2 = cropped_image(x, y, w, h, image_up)
    x, y, w, h = get_rectangle(mask1, image_up, 1)
    image_up_3 = cropped_image(x, y, w, h, image_up)
    # image_up = cv2.rectangle(image_up, (x, y), (x + w, y + h), (0, 255, 0))
    # cv2.imshow("up", image_up)
    
    image_down = cv2.bitwise_and(input_img, input_img, mask = mask2)
    image_bg_down = cv2.bitwise_and(image_bg, image_bg, mask = cv2.bitwise_not(mask2))
    image_down = image_down + image_bg_down
    
    x, y, w, h = get_rectangle(mask2, image_down, 1.1)
    image_down_1 = cropped_image(x, y, w, h, image_down)
    x, y, w, h = get_rectangle(mask2, image_down, 1.3)
    image_down_2 = cropped_image(x, y, w, h, image_down)
    x, y, w, h = get_rectangle(mask2, image_down, 1)
    image_down_3 = cropped_image(x, y, w, h, image_down)
    
    cv2.imwrite("cloth/up.jpg", image_up)
    cv2.imwrite("cloth/down.jpg", image_down)

    cv2.imwrite("cloth/up_1.jpg", image_up_1)
    cv2.imwrite("cloth/down_1.jpg", image_down_1)    
    
    cv2.imwrite("cloth/up_2.jpg", image_up_2)
    cv2.imwrite("cloth/down_2.jpg", image_down_2)   
         
    cv2.imwrite("cloth/up_3.jpg", image_up_3)
    cv2.imwrite("cloth/down_3.jpg", image_down_3)   
    
def cropped_image(x, y, w, h, img):
    new_image = np.ones((h, w, 3), np.uint8) * 255
    new_image[0 : h, 0 : w] = img[y : y + h, x : x + w]
    return new_image

def get_rectangle(mask1, img, factor):
         
    contours = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(contours) != 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        size = img.shape
        # print(size)
        # if size[0] > h * 1.6 and size[1] > w * 0.6:
        x = x - int(w * (factor - 1) / 2)
        y = y - int(h * (factor - 1) / 2)
        w = int(w * factor)
        h = int(h * factor)
        
        if x < 0:
            x = 0
            
        if y < 0:
            y = 0
            
        if x + w > size[1]:
            w = size[1] - x
        if h + y> size[0]:
            h = size[0] - y
        return [x, y, w, h]
    else:
        return [0, 0, 1, 1]