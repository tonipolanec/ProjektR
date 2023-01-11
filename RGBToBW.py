import os
import cv2

# Pretvara sve fotografije u boji iz foldera '.\dataset' u crno bijele fotografije te ih smiješta
# u isti folder s prefiksom 'bw_'

path = "dataset"
gray_path = "dataset_bw"

files = os.listdir(path)

for image in files:
   img = cv2.imread(os.path.join(path, image))
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   new_img_name = "bw_" + image
   cv2.imwrite(os.path.join(gray_path, new_img_name), gray)

