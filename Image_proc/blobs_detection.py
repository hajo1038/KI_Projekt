import cv2
import math
import numpy as np
import matplotlib.pylab as plt
from glob import glob


def main():
    blurred_image, image = read_image()
    image_with_lines = detect_lines(blurred_image)
    image_with_keypoints = detect_blobs(image_with_lines)
    show_window(image_with_keypoints, image)


def read_image():
    images_list = glob(r"../data/*.png")
    # image = cv2.imread(images_list[30])
    image = cv2.imread(r"..\data\IMG_4109.png")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return blurred_image, image


def detect_lines(blurred_image):
    # Copy edges to the images that will display the results in BGR
    edged = cv2.Canny(blurred_image, 50, 200, None, 3)
    #image_with_lines = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    image_with_lines = np.zeros((512, 512, 3), np.uint8)

    # Probabilistic Line Transform
    # linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 20, 3, 2, 10)

    treshold = 20
    minLineLength = 20
    maxLineGap = 5

    linesP = cv2.HoughLinesP(edged, 1, np.pi / 180, treshold, 3, minLineLength, maxLineGap)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            print(linesP[i])

            l = linesP[i][0]
            cv2.line(image_with_lines, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    print(len(linesP), "Lines were found.")
    return image_with_lines


def detect_blobs(image_with_lines):
    # Apply Laplacian of Gaussian
    blobs_log = cv2.Laplacian(image_with_lines, cv2.CV_64F)
    blobs_log = np.uint8(np.absolute(blobs_log))

    # Set up the detector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 100

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 1000

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.3
    params.maxCircularity = 0.9

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(blobs_log)
    print(len(keypoints), "Keypoints were found.")
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of the blob
    image_with_keypoints = cv2.drawKeypoints(image_with_lines, keypoints, np.array([]), (0, 255, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return image_with_keypoints


def show_window(image_with_keypoints, image):
    # Show keypoints
    winname1 = "Keypoints"
    cv2.namedWindow(winname1)
    cv2.imshow(winname1, image_with_keypoints)
    # cv2.imshow("cdstP", cdstP)
    cv2.imshow("Edged", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()