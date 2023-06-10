import cv2
import math
import numpy as np
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import os


def main():
    images_list = glob(r"../data/*.png")
    df = open_csv()
    for image in images_list:
        head, tail = os.path.split(image)
        image = cv2.imread(image)
        blurred_image, image = preprocess_image(image)
        area_of_all_contours, number_of_contours = detect_contour(image, blurred_image)
        print("Area of all Contours " + str(area_of_all_contours))
        print("Number of Contours " + str(number_of_contours))
        number_of_lines = detect_lines(image)
        print("Number of Lines " + str(number_of_lines))

        row_number = df[df.file == tail].index
        print(tail)
        df.loc[df.index[row_number], "Contours"] = number_of_contours
        df.loc[df.index[row_number], "Lines"] = number_of_lines
        df.loc[df.index[row_number], "Contours Size"] = area_of_all_contours
    df.to_csv("data_with_features.csv", encoding='utf-8', index=False)


def open_csv():
    df = pd.read_csv(r"../Labeling_App/Labels.csv")
    df["Lines"] = ""
    df["Contours"] = ""
    df["Contours Size"] = ""
    return df


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


def detect_lines(image):
    image = auto_canny_edge_detection(image)
    # Probabilistic Line Transform
    image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, 20, 3, 2, 10)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(image_with_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    try:
        length = len(linesP)
        return length
    except TypeError:
        return 0


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    return blurred_image, image


def detect_contour(image, blurred_image):
    edged = cv2.Canny(blurred_image, 10, 100)
    auto_edge = auto_canny_edge_detection(blurred_image)

    # define a (3, 3) structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # apply the dilation operation to the edged image
    dilate = cv2.dilate(edged, kernel, iterations=1)
    erosion = cv2.erode(dilate, kernel, iterations=1)

    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(erosion)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 750

    # output image with only the kept components
    result_image = np.zeros_like(im_with_separated_blobs)
    area_of_all_contours = 0
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            result_image[im_with_separated_blobs == blob + 1] = 255
            area_of_all_contours = area_of_all_contours + sizes[blob]
            #print(sizes[blob])

    result_image = result_image.astype(np.uint8)
    # find the contours in the dilated image
    contours, _ = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    return area_of_all_contours, len(contours)


if __name__ == '__main__':
    main()