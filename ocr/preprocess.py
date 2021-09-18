from pathlib import Path

import cv2
import pytesseract
import numpy as np


class PreprocessImage:
    def __init__(self, img, res=None, res_scale=None, disp_res_scale=0.25):
        if res is None and res_scale is None:
            res_scale = 1

        if res is None:
            res = (int(img.shape[1] * res_scale), int(img.shape[0] * res_scale))

        disp_res = (int(img.shape[1] * disp_res_scale), int(img.shape[0] * disp_res_scale))
        self.disp_res = disp_res

        print("res:", res)
        self.img = cv2.resize(img, res)

    def get_disp_img(self):
        print(self.disp_res)
        return cv2.resize(self.img, self.disp_res)

    def preprocess_img(self):
        processed = self.process_color(self.img)
        print("after color", processed.shape)
        processed = self.denoise(processed)
        _, processed = self.threshold(processed)

        self.img = processed
        return processed

    def process_color(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def denoise(self, img):
        img = cv2.medianBlur(img, 5)
        return img

    def threshold(self, img):
        # img = cv2.convertScaleAbs(self.img, )
        ret, thresh = cv2.threshold(
            img, 0, 255, cv2.THRESH_OTSU
        )
        return ret, thresh


dataset_dir = Path("./ocr-datasets")

print("imgs: ", list(dataset_dir.iterdir()))
img_path = dataset_dir / "paper-on-laptop.jpg"
img = cv2.imread(str(img_path))

preprocessImage = PreprocessImage(img, res_scale=0.25)
img = preprocessImage.preprocess_img()
img = preprocessImage.get_disp_img()

cv2.imshow("img", img)
cv2.waitKey(0)
while True:
    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
        break
