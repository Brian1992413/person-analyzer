import age_gender.predict_bbox as age_gender_detect
import skin.detect_skin as skin_detect
import asyncio
import cv2
import numpy as np
import cloth.hugging_face_detect as segment_cloth
from fastai.vision.all import *
from PIL import Image
import time
def get_x(r):
    return 'images_original/' + r['image']

def get_y(r):
    return r['label_cat'].split(' ')

def convert_palette(color):
    color_value = ["373028", "422811", "513b2e", "6f503c", "81654f", "9d7a54", "bea07e", "e5c8a6", "e7c1b8", "f3dad6", "fbf2f3"]
    str_value = ["Black", "Black", "Brown", "Brown", "Olive", "Olive", "Medium", "Medium", "Fair", "Fair", "Very Fair", "Very Fair"]
    color = color.lstrip("#")
    for i in range(10):
        if int(color, 16) <= int(color_value[0], 16):
            return str_value[0]
        if int(color, 16) >= int(color_value[10], 16):
            return str_value[11]
        if int(color_value[i], 16) <= int(color, 16) and int(color_value[i + 1], 16) >= int(color, 16):
            return str_value[i + 1]
    return "None"

async def main():
    x = time.time()
    result = age_gender_detect.detect_age_gender(input_image)
    print("age_gender_race : " + str(time.time() - x))
    
    age = result["age"]
    gender = result["gender"]
    nationality = result["race"]
    
    x = time.time()
    # skin_tone = skin_detect.detect_skin() # this is the part to detect the skin. yuo can enable after that.
    skin_tone = "Yellow"
    print("Skin detect : " + str(time.time() - x))
    
    detect_info = []
    detect_info.append("Gender : " + gender)
    detect_info.append("Age : " + age)
    detect_info.append("Nationality : " + nationality)
    # detect_info.append("Skin : " + convert_palette(skin_tone))
    detect_info.append("Skin : " + skin_tone)
    return detect_info

input_image = "source/14.png"
output_face_image = "result/face.jpg"
output_image = "result/result.jpg"
output_dst_image = "cloth/mask.jpg"

def detect_face_cv(image_path, default_max_size=800, size = 300, padding = 0.25):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_face_cascade = cv2.CascadeClassifier('age_gender/haarcascade_frontalface_alt.xml')
    faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    # print(faces)
    (x, y, w, h) = faces[0]
    new_image = np.zeros((h, w, 3), np.uint8)
    # print(h, w)
    new_image[0 : h, 0 : w] = img[y : y + h, x : x + w]
    size = new_image.shape
    new_image = cv2.resize(new_image, (300, 300))
    cv2.imwrite("result/face.jpg", new_image)

def pred_cloth(img_path, IsUp):
    img = PILImage.create(img_path)
    # img = Image.open(img_path)
    if IsUp == True:
        pred , _ , _ = learn_up.predict(img)
    else:
        pred , _ , _ = learn_down.predict(img)
    
    if len(pred) == 1:
        results = pred[0]
        gender = result[0]
        if results == 'Skirts' and gender[-4: -1] == "Mal":
            results = 'Shorts'
        elif results == 'Skirts' and gender[-4:-1] == "mal":
            img_PIL = Image.open("cloth/down_3.jpg")
            if img_PIL.height < img_PIL.width:
                results = 'Shorts'
        elif results == 'Shorts':
            img_PIL = Image.open("cloth/down_1.jpg")
            if img_PIL.height > img_PIL.width * 1.2:
                results = 'Pants'
        # if results == 'Shorts' or results == 'Pants':
        #     pred1,_,_ = learn.predict(img)
        #     print("Other result is " + pred1[0])
            # result(len(pred1))
        return results
    else:
        return 'Not Sure'

if __name__ == "__main__":
    start_time = time.time()
    x = time.time()
    detect_face_cv(input_image)
    print("Face detect : " + str(time.time() - x))
    
    result = asyncio.run(main())
    segment_cloth.cloth_segment(input_image)
    # cloth_up = class_cloth.detect_type_clothes('cloth/up.jpg')
    # cloth_down = class_cloth.detect_type_clothes('cloth/down.jpg')
    # learn = load_learner('cloth/cloth_classification.pkl')
    learn_up = load_learner('cloth/cloth_classification_up.pkl')
    learn_down = load_learner('cloth/cloth_classification_down.pkl')
    IsDress = "False"
    if pred_cloth('cloth/down_1.jpg', IsUp = True) == 'Dresses':
        IsDress = "True"
        cloth_up = pred_cloth('cloth/up.jpg', IsUp = True)
        cloth_down = "Dresses"
        if cloth_up == cloth_down:
            cloth_up = ""
    else:
        cloth_up = pred_cloth('cloth/up.jpg', IsUp = True)
        if cloth_up == 'Not Sure' or cloth_up == 'Dresses':
            cloth_up = pred_cloth('cloth/up_2.jpg', IsUp = True)
        if cloth_up == 'Not Sure' or cloth_up == 'Dresses':
            cloth_up = pred_cloth('cloth/up_1.jpg', IsUp = True)
        cloth_down = pred_cloth('cloth/down.jpg', IsUp = False)
        if cloth_down == 'Not Sure':
            cloth_down = pred_cloth('cloth/down_2.jpg', IsUp = False)
        if cloth_down == 'Not Sure':
            cloth_down = pred_cloth('cloth/down_1.jpg', IsUp = False)
    
    # Load the image
    image = cv2.imread(output_face_image)  # Replace "path_to_your_image.jpg" with the actual path to your image
    image_dst = cv2.imread(output_dst_image)
    # Define padding and text details
    padding = 400
    
    image_origin = cv2.imread(input_image)
    desired_width = 300 + padding
    desired_width = int(desired_width / 2)
    height, width = image_origin.shape[:2]
    aspect_ratio = desired_width / width

    # Calculate the new width
    new_height_origin = int(height * aspect_ratio)
    image_origin = cv2.resize(image_origin, (desired_width, new_height_origin))
    image_dst = cv2.resize(image_dst, (desired_width, new_height_origin))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 1
    text_color = (0, 0, 0)  # Black color, you can customize as needed

    # Get image dimensions
    height, width = image.shape[:2]

    # Create a new image with padding
    new_width = width + padding
    new_height = height + new_height_origin
    new_image = np.ones((new_height, new_width, 3), np.uint8) * 255  # White background, you can customize as needed
    
    # Paste the original image onto the new image
    new_image[0:height, 0:width] = image
    new_image[height : height + new_height_origin, 0 : desired_width] = image_origin
    new_image[height : height + new_height_origin, desired_width : desired_width * 2] = image_dst
    
    # Add text to the image
    (text_width, text_height), _ = cv2.getTextSize("nationality : middle east_more letters", font, font_scale, font_thickness)
    text_position = ((new_width - text_width) // 2, (new_height + text_height) // 2)
    cv2.putText(new_image, result[0], (320, 50), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(new_image, result[1], (320, 100), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(new_image, result[2], (320, 150), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    # cv2.putText(new_image, result[3], (320, 172), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    cv2.putText(new_image, "IsDress : " + IsDress, (320, 200), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    if IsDress: cv2.putText(new_image, "Cloth Type : " + cloth_up + ", " + cloth_down, (320, 250), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    else: cv2.putText(new_image, "Cloth Type : " + cloth_up + "" + cloth_down, (320, 250), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    # Save or display the modified image
    cv2.imwrite(output_image, new_image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Running time : " + str(elapsed_time))
    cv2.imshow("Modified Image", new_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()