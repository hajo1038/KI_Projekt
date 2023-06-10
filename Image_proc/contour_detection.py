import cv2
import math
import numpy as np
import matplotlib.pylab as plt
from glob import glob


def main():
    blurred_image, image = read_image()
    result_image, image_copy = detect_contour(image, blurred_image)
    show_image(result_image, image_copy)


def read_image():
    images_list = glob(r"../data/*.png")
    image = cv2.imread(images_list[30])
    print(images_list[30])
    #image = cv2.imread(r"..\data\IMG_3850.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    return blurred_image, image


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0 - sigma) * md))
    upper_value = int(min(255, (1.0 + sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


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
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            result_image[im_with_separated_blobs == blob + 1] = 255
            print(sizes[blob])

    result_image = result_image.astype(np.uint8)
    # find the contours in the dilated image
    contours, _ = cv2.findContours(result_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")
    return result_image, image_copy


def show_image(result_image, image_copy):
    # cv2.imshow("Dilated image", dilate)
    cv2.imshow("Blobs removed", result_image)
    cv2.imshow("contours", image_copy)
    # cv2.imshow("blurred", blurred)
    # cv2.imshow("Edged", edged)
    # cv2.imshow("Erosion", erosion)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()