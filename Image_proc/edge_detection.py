import cv2
import math
import numpy as np
import matplotlib.pylab as plt
from glob import glob


def main():
    blurred_image, image = read_image()
    auto_edged_image = auto_canny_edge_detection(blurred_image)
    image_with_lines = detect_lines(auto_edged_image)
    show_images(image, image_with_lines)


def read_image():
    images_list = glob(r"../data/*.png")
    # image = cv2.imread(images_list[30])
    image = cv2.imread(r"..\data\IMG_3850.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image, image


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


def detect_lines(image):
    # Probabilistic Line Transform
    image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 20, 3, 2, 10)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image_with_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    return image_with_lines


def show_images(image1, image2):
    # for showing two images in one window
    combined_image = np.concatenate((image1, image2), axis=1)
    window_name = "Kantenerkennung"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, 40, 30)
    cv2.imshow(window_name, combined_image)
    # cv2.imshow("cdstP", image_with_lines)
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", auto_edge)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return


if __name__ == '__main__':
    main()