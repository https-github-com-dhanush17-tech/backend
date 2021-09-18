from pathlib import Path

import cv2
import pytesseract
import numpy as np

from backend.ocr.preprocess import PreprocessImage

dataset_dir = Path("./ocr-datasets")
tesseract_config = r"--oem 2 --psm 3"

print("imgs: ", list(dataset_dir.iterdir()))
img_path = dataset_dir / "40.jpg"
# img_path = dataset_dir / "preprocessed-receipt.jpg"
print("img:", img_path)
img = cv2.imread(str(img_path))
print("img res:", img.shape)

preprocessImage = PreprocessImage(img)
processed_img = preprocessImage.preprocess_img()
print("processed_img res:", processed_img.shape)

ocr_output = pytesseract.image_to_string(processed_img, config=tesseract_config)
print("===============\n\n\nText detected:\n", ocr_output)

cv2.imshow("img", cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25))))
cv2.imshow("processed_img", cv2.resize(processed_img, (int(processed_img.shape[1] * 0.25), int(processed_img.shape[0] * 0.25))))
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
