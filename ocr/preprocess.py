from pathlib import Path

import cv2
import pytesseract
import numpy as np


class PreprocessImage:
    def __init__(self, img, res_scale=0.25):
        res = (
            int(img.shape[1] * res_scale),
            int(img.shape[0] * res_scale),
        )

        self.res = res
        self.img = cv2.resize(img, self.res)

    def preprocess_img(self):
        img = self.img

        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.medianBlur(img, 5)
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        # smoothed = self.smoother_edges(blurred, (7, 7), (1, 1))
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

        median = np.median(gray)
        sigma = 0.33
        lower_thresh = int(max(0, (1.0 - sigma) * median))
        upper_thresh = int(min(255, (1.0 + sigma) * median))
        print("canny thresh:", lower_thresh, upper_thresh)
        canny = cv2.Canny(thresh, lower_thresh, upper_thresh)

        # canny = self.dilate(canny)
        # canny = self.close(canny)
        # canny = self.erode(canny)
        # canny = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
        # canny = cv2.GaussianBlur(canny, (3, 3), 0)
        kernel = np.ones((5, 5), np.uint8)
        canny = cv2.dilate(canny, kernel, iterations=2)
        canny = cv2.erode(canny, kernel, iterations=1)

        contours, hierarchy = cv2.findContours(
            canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = self.filter_contours_area(contours, 9000)
        # rectangles = self.filter_contours_closed(contours, hierarchy)
        rectangles = self.get_contours_of_shape(contours, 4)

        if len(rectangles) != 0:
            rectangle_contour = self.get_largest_contour(rectangles)

            perimeter = cv2.arcLength(rectangle_contour, True)
            rectangle = cv2.approxPolyDP(rectangle_contour, 0.10 * perimeter, True)

            dewarped = self.unwarp_rect(img, rectangle)
            print("rectangle:", rectangle)
            cv2.imshow("dwarmped", dewarped)

        drawing = self.img.copy()
        cv2.drawContours(drawing, contours, -1, (0, 255, 0), 1)

        if len(rectangles) != 0:
            cv2.drawContours(drawing, [rectangle], 0, (255, 0, 0), 3)

        # Hough Lines
        # from https://stackoverflow.com/a/45560545
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 1  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 100  # minimum number of pixels making up a line
        max_line_gap = 20  # maximum gap in pixels between connectable line segments
        line_image = np.copy(img) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(
            canny, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
        )

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        drawing = cv2.addWeighted(drawing, 0.8, line_image, 1, 0)

        # gray = self.process_color(self.img)
        # denoised = self.denoise(gray)
        # _, thresh = self.threshold(denoised)
        # opened = self.opening(thresh)
        # canny = self.canny(opened)

        cv2.imshow("blurred", blurred)
        # cv2.imshow("smoothed", smoothed)
        cv2.imshow("thresh", thresh)
        cv2.imshow("canny", canny)

        return drawing

    def process_color(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def denoise(self, img):
        img = cv2.medianBlur(img, 3)
        return img

    def threshold(self, img):
        # img = cv2.convertScaleAbs(self.img, )
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        return ret, thresh

    def dilate(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)

    def erode(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)

    def open(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)

    def close(self, img):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel)

    def canny(self, img):
        return cv2.Canny(img, 100, 200)

    def unsharp_mask(self, img, blur_size, img_weight, gaussian_weight):
        # code from https://stackoverflow.com/questions/42872353/correcting-rough-edges/42872732
        gaussian = cv2.GaussianBlur(img, blur_size, 0)
        return cv2.addWeighted(img, img_weight, gaussian, gaussian_weight, 0)

    def smoother_edges(
        self,
        img,
        first_blur_size,
        second_blur_size=(5, 5),
        img_weight=1.5,
        gaussian_weight=-0.5,
    ):
        # code from https://stackoverflow.com/questions/42872353/correcting-rough-edges/42872732
        # blur the image before unsharp masking
        img = cv2.GaussianBlur(img, first_blur_size, 0)
        # perform unsharp masking
        return self.unsharp_mask(img, second_blur_size, img_weight, gaussian_weight)

    def filter_contours_area(self, contours, area_thresh, keep_big=True):
        filtered = []
        for contour in contours:
            print("contour area:", cv2.contourArea(contour))
            if keep_big and cv2.contourArea(contour) > area_thresh:
                filtered.append(contour)
            elif not keep_big and cv2.contourArea(contour) < area_thresh:
                filtered.append(contour)

        return filtered

    def detect_shape(self, contour, num_sides):
        """Returns True if shape with provided number of sides is detected with supplied closed contour and is convex, False otherwise."""
        # must be closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.10 * perimeter, True)

        if len(approx) == num_sides and cv2.isContourConvex(approx):
            return True

        return False

    def get_contours_of_shape(self, contours, num_sides):
        new_contours = []
        for contour in contours:
            if self.detect_shape(contour, num_sides):
                new_contours.append(contour)

        return new_contours

    def get_largest_contour(self, contours):
        areas = [cv2.contourArea(contour) for contour in contours]
        return contours[np.argmax(areas)]

    def filter_contours_closed(self, contours, hierarchy):
        # TODO need to fix to make this work
        filtered = []
        for contour, h in zip(contours, hierarchy):
            print(h)
            if cv2.isContourConvex(contour) and h[2] != -1:
                filtered.append(contour)

        return filtered

    def sort_rect_points(self, points_new):
        points_old = np.array(points_new).reshape(4, 2)
        points_new = np.zeros((4, 1, 2), dtype=np.int32)
        sum = points_old.sum(1)

        points_new[0] = points_old[np.argmin(sum)]
        points_new[3] = points_old[np.argmax(sum)]
        diff = np.diff(points_old, axis=1)
        points_new[1] = points_old[np.argmin(diff)]
        points_new[2] = points_old[np.argmax(diff)]
        return points_new

    def unwarp_rect(self, img, rect_points):
        rect_points = self.sort_rect_points(rect_points).astype(np.float32)
        corner_points = np.array(
            [
                [0, 0],
                [self.img.shape[1], 0],
                [0, self.img.shape[0]],
                [self.img.shape[1], self.img.shape[0]]
            ],
            dtype=np.float32
        )

        trans_mat = cv2.getPerspectiveTransform(rect_points, corner_points)
        return cv2.warpPerspective(
            img, trans_mat, (self.img.shape[1], self.img.shape[0])
        )


if __name__ == "__main__":
    live_cam = False
    if live_cam:
        cap = cv2.VideoCapture(0)
    else:
        dataset_dir = Path("./ocr-datasets")
        img_path = dataset_dir / "ex-0.jpg"
        img = cv2.imread(str(img_path))

    # print("imgs: ", list(dataset_dir.iterdir()))
    # img_path = dataset_dir / "textbook-white-background.jpg"
    # img = cv2.imread(str(img_path))
    print("img loaded:", img)

    # preprocessImage = PreprocessImage(img, disp_res_scale=0.25)
    while True:
        if live_cam:
            ret, img = cap.read()
            if not ret:
                break

        print("org res", img.shape)

        preprocessImage = PreprocessImage(img, res_scale=0.25)
        processed_img = preprocessImage.preprocess_img()

        cv2.imshow("img", preprocessImage.img)
        cv2.imshow("processed", processed_img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
