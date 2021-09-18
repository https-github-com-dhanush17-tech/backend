from pathlib import Path

import cv2
import pytesseract
import numpy as np

from backend.ocr.preprocess import PreprocessImage


class OCR:
    def __init__(self):
        self.tesseract_config = r"--oem 2 --psm 3"

    def get_text(self, img):
        preprocessImage = PreprocessImage(img)
        processed_img = preprocessImage.preprocess_img()

        ocr_output = pytesseract.image_to_string(
            processed_img, config=self.tesseract_config
        )

        return ocr_output


if __name__ == "__main__":
    dataset_dir = Path("./ocr-datasets")
    img_path = dataset_dir / "42.jpg"
    img = cv2.imread(str(img_path))

    print("imgs: ", list(dataset_dir.iterdir()))
    # img_path = dataset_dir / "preprocessed-receipt.jpg"
    print("img:", img_path)
    print("img res:", img.shape)

    ocr = OCR()
    ocr_output = ocr.get_text(img)

    print("===============\n\n\nText detected:\n", ocr_output)
    cv2.imshow(
        "img", cv2.resize(img, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)))
    )

    k = cv2.waitKey(0) & 0xFF
    if k == 27:
        cv2.destroyAllWindows()
    # preprocessImage = PreprocessImage(img)
    # processed_img = preprocessImage.preprocess_img()
    # print("processed_img res:", processed_img.shape)
    #
    # tesseract_config = r"--oem 2 --psm 3"
    # ocr_output = pytesseract.image_to_string(processed_img, config=tesseract_config)
    # print("===============\n\n\nText detected:\n", ocr_output)

    # cv2.imshow("processed_img", cv2.resize(processed_img, (int(processed_img.shape[1] * 0.25), int(processed_img.shape[0] * 0.25))))
