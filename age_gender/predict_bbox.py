from __future__ import print_function, division
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import cv2

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def reverse_resized_rect(rect,resize_ratio):
    l = int(rect.left() / resize_ratio)
    t = int(rect.top() / resize_ratio)
    r = int(rect.right() / resize_ratio)
    b = int(rect.bottom() / resize_ratio)
    new_rect = dlib.rectangle(l,t,r,b)

    return [l,t,r,b] , new_rect
    
def detect_face(image_paths, default_max_size=800, size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('age_gender/dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('age_gender/dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    rects = []
    for index, image_path in enumerate(image_paths):
        if index % 1000 == 0:
            # print('---%d/%d---' %(index, len(image_paths)))
            pass

        img = dlib.load_rgb_image(image_path)

        old_height, old_width, _ = img.shape
        if old_width > old_height:
            resize_ratio = default_max_size / old_width
            new_width, new_height = default_max_size, int(old_height * resize_ratio)
        else:
            resize_ratio = default_max_size / old_height
            new_width, new_height =  int(old_width * resize_ratio), default_max_size
        img = dlib.resize_image(img, cols=new_width, rows=new_height)

        dets = cnn_face_detector(img, 1)
        num_faces = len(dets)
        if num_faces == 0:
            # print("Sorry, there were no faces found in '{}' with this method, I will try other method".format(image_path))
            # return None
            continue
        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()

        for detection in dets:
            rect = detection.rect
            faces.append(sp(img, rect))
            rect_tpl ,rect_in_origin = reverse_resized_rect(rect,resize_ratio)
            rects.append(rect_in_origin)
        images = dlib.get_face_chips(img, faces, size=size, padding = padding)
        for idx, image in enumerate(images):
            img_name = image_path.split("/")[-1]
            path_sp = img_name.split(".")
            face_name = "result/face.jpg"
            dlib.save_image(image, face_name) 

    return rects

def predidct_age_gender_race(save_prediction_at, imgs_path = 'cropped_faces/'):
    img_names = ["result/face.jpg"]
    device = torch.device('cpu')

    model_fair_7 = torchvision.models.resnet34(pretrained=True)
    model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
    model_fair_7.load_state_dict(torch.load('age_gender/res34_fair_align_multi_7_20190809.pt', map_location=torch.device('cpu')))
    #model_fair_7.load_state_dict(torch.load('fair_face_models/res34_fair_align_multi_7_20190809.pt'))
    model_fair_7 = model_fair_7.to(device)
    model_fair_7.eval()

    model_fair_4 = torchvision.models.resnet34(pretrained=True)
    model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
    model_fair_4.load_state_dict(torch.load('age_gender/res34_fair_align_multi_4_20190809.pt', map_location=torch.device('cpu')))
    model_fair_4 = model_fair_4.to(device)
    model_fair_4.eval()

    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # img pth of face images
    face_names = []
    # list within a list. Each sublist contains scores for all races. Take max for predicted race
    race_scores_fair = []
    gender_scores_fair = []
    age_scores_fair = []
    race_preds_fair = []
    gender_preds_fair = []
    age_preds_fair = []
    race_scores_fair_4 = []
    race_preds_fair_4 = []

    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            # print("Predicting... {}/{}".format(index, len(img_names)))
            pass
        

        face_names.append(img_name)
        image = dlib.load_rgb_image(img_name)
        image = trans(image)
        image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
        image = image.to(device)

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        gender_preds_fair.append(gender_pred)
        age_preds_fair.append(age_pred)

        # fair 4 class
        outputs = model_fair_4(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:4]
        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        race_pred = np.argmax(race_score)

        race_scores_fair_4.append(race_score)
        race_preds_fair_4.append(race_pred)

    result_dict = {}
    if (race_preds_fair[0] == 0): result_dict["race"] = "White"
    elif (race_preds_fair[0] == 1): result_dict["race"] = "Black"
    elif (race_preds_fair[0] == 2): result_dict["race"] = "Latino_Hispanic"
    elif (race_preds_fair[0] == 3): result_dict["race"] = "East Asian"
    elif (race_preds_fair[0] == 4): result_dict["race"] = "Southeast Asian"
    elif (race_preds_fair[0] == 5): result_dict["race"] = "Indian"
    elif (race_preds_fair[0] == 6): result_dict["race"] = "Middle Eastern"
    
    # race fair 4
    if (race_preds_fair_4[0] == 0): result_dict["race4"] = "White"
    elif (race_preds_fair_4[0] == 1): result_dict["race4"] = "Black"
    elif (race_preds_fair_4[0] == 2): result_dict["race4"] = "Asian"
    elif (race_preds_fair_4[0] == 3): result_dict["race4"] = "Indian"

    # gender
    if (gender_preds_fair[0] == 0): result_dict["gender"] = "Male"
    elif (gender_preds_fair[0] == 1): result_dict["gender"] = "Female"
    
    if (age_preds_fair[0] == 0): result_dict["age"] = "0-2"
    elif (age_preds_fair[0] == 1): result_dict["age"] = "3-9"
    elif (age_preds_fair[0] == 2): result_dict["age"] = "10-19"
    elif (age_preds_fair[0] == 3): result_dict["age"] = "20-29"
    elif (age_preds_fair[0] == 4): result_dict["age"] = "30-39"
    elif (age_preds_fair[0] == 5): result_dict["age"] = "40-49"
    elif (age_preds_fair[0] == 6): result_dict["age"] = "50-59"
    elif (age_preds_fair[0] == 7): result_dict["age"] = "60-69"
    elif (age_preds_fair[0] == 8): result_dict["age"] = "70+"
    
    return result_dict

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_box(rect, img_path, result):
    image = cv2.imread(img_path[0])
    bbox = rect_to_bb(rect[0])
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
    race = result["race"]
    age = result["age"]
    gender = result["gender"]
    text_gender = "Gender : " + gender
    text_age = "Age : " + age
    text_nation = "Nationality : " + race
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)
    thickness = 2
    
    # text_size, _ = cv2.getTextSize(text_gender, font, font_scale, thickness)
    # print(text_gender)
    # print(text_age)
    # print(text_nation)
    cv2.putText(image, text_gender, (bbox[0], bbox[1] + bbox[3] + 30), font, font_scale, color, thickness)
    cv2.putText(image, text_age, (bbox[0], bbox[1] + bbox[3] + 60), font, font_scale, color, thickness)
    cv2.putText(image, text_nation, (bbox[0], bbox[1] + bbox[3] + 90), font, font_scale, color, thickness)
    
    cv2.imwrite('result/result.jpg', image)
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_age_gender(input_image):
    imgs = [input_image]
    # bboxes = detect_face(imgs)
    result = predidct_age_gender_race("test_outputs.csv")
    return result