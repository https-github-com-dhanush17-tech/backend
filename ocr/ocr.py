# TODO use confidence for each word/letter if possible
from pathlib import Path

import cv2
import pytesseract
import numpy as np
from autocorrect import Speller

from .preprocess import PreprocessImage


class OCR:
    def __init__(self, conf_thresh=0.0):
        self.tesseract_config = r"--oem 2 --psm 3"
        self.conf_thresh = conf_thresh
        self.spell_cleanup = Speller(only_replacements=False)

    def get_text(self, img):
        preprocessImage = PreprocessImage(img)
        processed_img = preprocessImage.preprocess_img()

        ocr_output = pytesseract.image_to_string(
            processed_img, config=self.tesseract_config
        )

        ocr_data = pytesseract.image_to_data(
            processed_img,
            config=self.tesseract_config,
            output_type=pytesseract.Output.DICT,
        )

        ocr_conf_output = ""
        for conf, word in zip(ocr_data["conf"], ocr_data["text"]):
            if int(conf) >= self.conf_thresh * 100:
                corrected = self.spell_cleanup(word)
                if corrected != word and corrected != "":
                    ocr_conf_output += corrected
                else:
                    ocr_conf_output += word
                ocr_conf_output += " "
            if int(conf) == -1:
                ocr_conf_output += "\n"

        return ocr_conf_output


if __name__ == "__main__":
    dataset_dir = Path("./ocr-datasets")
    img_path = dataset_dir / "42.jpg"
    img = cv2.imread(str(img_path))

    # print("imgs: ", list(dataset_dir.iterdir()))
    # img_path = dataset_dir / "preprocessed-receipt.jpg"
    # print("img:", img_path)
    # print("img res:", img.shape)

    ocr = OCR()
    ocr_output = ocr.get_text(img)

    # print("===============\n\n\nText detected:\n", ocr_output)
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
