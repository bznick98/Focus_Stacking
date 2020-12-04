import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys

intro = \
"""
Focus Stacking

This project is the final project of CS445 Fall2020,
Team: Zongnan Bao(zb3) and Han Chen(hanc3)\n\n
"""


def align_images(images):
    """
    Align input images using key-points extraction and homography matrix
    @input: array of images
    @output: array of aligned images
    """
    # use the first image as the reference image
    ref_img = images[0]
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # output array
    aligned_images = [ref_img,]

    # find homography between other images and ref img
    for img in images[1:]:
        # convert it to grayscale for feature extraction
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # use SIFT or ORB feature detection, or KAZE, seems more stable than ORB
        max_features = 1000
        # sift = cv2.xfeatures2d.SIFT_create()
        detector = cv2.KAZE_create(max_features)

        # find keypoints and descriptors
        kp_a, des_a = detector.detectAndCompute(img_gray, None)
        kp_b, des_b = detector.detectAndCompute(ref_gray, None)

        # Matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_a, des_b, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        numMatches = int(len(good)) 

        matches = good

        imMatches = cv2.drawMatches(ref_img, kp_a, img, kp_b, matches, None)
        cv2.imwrite("matches.jpg", imMatches)

        # extract location of good matches
        pts_a = np.zeros((numMatches, 2), dtype=np.float32)
        pts_b = np.zeros((numMatches, 2), dtype=np.float32)

        for idx, match in enumerate(matches):
            pts_a[idx, :] = kp_a[match.queryIdx].pt 
            pts_b[idx, :] = kp_b[match.trainIdx].pt

        H, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC)

        # transform img
        height, width, channels = img.shape
        img_warped = cv2.warpPerspective(img, H, (width, height))
        aligned_images.append(img_warped)

    return aligned_images


# find the largest rectanlge in an 2d-matrix
# source: https://stackoverflow.com/questions/38277859/obtain-location-of-largest-rectangle
from collections import namedtuple

Info = namedtuple('Info', 'start height')

# returns height, width, and position of the top left corner of the largest
#  rectangle with the given value in mat
def max_size(mat, value=0):
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size_start, start_row = max_rectangle_size(hist), 0
    for i, row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        mss = max_rectangle_size(hist)
        if area(mss) > area(max_size_start):
            max_size_start, start_row = mss, i+2-mss[0]
    return max_size_start[:2], (start_row, max_size_start[2])

# returns height, width, and start column of the largest rectangle that
#  fits entirely under the histogram
def max_rectangle_size(histogram):
    stack = []
    top = lambda: stack[-1]
    max_size_start = (0, 0, 0) # height, width, start of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                max_size_start = max(
                    max_size_start,
                    (top().height, pos - top().start, top().start),
                    key=area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here

    pos += 1
    for start, height in stack:
        max_size_start = max(max_size_start, (height, pos - start, start),
            key=area)

    return max_size_start

def area(size): return size[0]*size[1]

# focus_stacking
def focus_stacking(images):
    """
    achieves the functionality of focus stacking
    @input: array of images
    @output: single image that stacked the depth of fields of all images
    """
    # 1 - align images
    aligned_images = align_images(images)

    # display alignment
    fg, axs = plt.subplots(1,2)
    fg.suptitle('Reference Image and the Second Image(Aligned)')
    axs[0].imshow(images[0][:,:,[2,1,0]])
    axs[1].imshow(aligned_images[1][:,:,[2,1,0]])
    plt.show()

    # convert to grayscale to get largest rectangle & Laplacian of Gaussian
    aligned_gray = [cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY) for aligned_img in aligned_images]

    # convert to binary map to perform max-size rectangle
    aligned_bin, thre = cv2.threshold(aligned_gray[1], 0, 1, cv2.THRESH_BINARY)
    plt.imshow(aligned_gray[1])
    plt.show()
    plt.imshow(thre)
    plt.show()

    # # crop image so that it fits in the largest rectangle in warped image
    # for i in range(len(aligned_gray)):


    # 2 - Gaussian blur on all images
    for i in range(len(aligned_gray)):
        aligned_gray[i] = cv2.GaussianBlur(aligned_gray[i], ksize=(3, 3), sigmaX=3)

    # 3 - Laplacian on blurred images
    for i in range(len(aligned_gray)):
        aligned_gray[i] = cv2.Laplacian(aligned_gray[i], cv2.CV_64F, ksize=3)

    # display LoG
    fg, axs = plt.subplots(1,2)
    fg.suptitle('Laplacian of Gaussian(edge detect) for all images')
    axs[0].imshow(aligned_gray[0][:,:])
    axs[1].imshow(aligned_gray[1][:,:])
    plt.show()

    # prepare output image
    canvas = np.zeros(images[0].shape)

    # get the maxmimum value of LoG across all images
    max_LoG = np.max(np.absolute(aligned_gray), axis=0)

    # find mask corresponding to maximum (which LoG ahiceves the maximum)
    masks = (np.absolute(aligned_gray) == max_LoG).astype('uint8')
    
    # apply masks
    for i in range(len(aligned_images)):
        canvas = cv2.bitwise_not(aligned_images[i], canvas, mask=masks[i])

    # process the output
    canvas = (255-canvas)
    return canvas


if __name__ == "__main__":
    # print introdutory text
    print(intro)

    # load images
    file_names = sys.argv[1:]
    num_files = len(file_names)
    
    # check number of files
    assert num_files > 1, "Provide at least 2 images."

    # load images
    images = np.array([cv2.imread(f_name) for f_name in file_names])
    
    # display original images
    fg, axs = plt.subplots(1,2)
    fg.suptitle('Unprocessed Images')
    axs[0].imshow(images[0][:,:,[2,1,0]])
    axs[1].imshow(images[1][:,:,[2,1,0]])
    plt.show()

    # TODO: check the filenames are valid

    # focus stacking
    canvas = focus_stacking(images)
    plt.imshow(canvas[:,:,[2,1,0]])
    plt.show()

    # write to file
    cv2.imwrite('output.jpg', canvas)