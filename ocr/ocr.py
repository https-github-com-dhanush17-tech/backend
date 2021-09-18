from pathlib import Path

import cv2
import pytesseract
import numpy as np

from backend.ocr.preprocess import PreprocessImage

dataset_dir = Path("./ocr-datasets")
tesseract_config = r"--oem 3 --psm 6"

print("imgs: ", list(dataset_dir.iterdir()))
img_path = dataset_dir / "paper-on-laptop.jpg"
# img_path = dataset_dir / "preprocessed-receipt.jpg"
print("img:", img_path)
img = cv2.imread(str(img_path))

preprocessImage = PreprocessImage(img)
processed_img = preprocessImage.preprocess_img()

ocr_output = pytesseract.image_to_string(processed_img, config=tesseract_config)
print("Text detected:\n", ocr_output)

cv2.imshow("img", img)
cv2.waitKey(-1)
