import numpy as np
import cv2
import matplotlib.pyplot as plt

# These are all helper functions used by focus

def align_images(images):
    """
    Align input images using key-points extraction and homography matrix
    @input: array of images
    @output: array of aligned images
    """
    # determine input image is gray or not
    isColor = True
    if len(images.shape) <= 3:
         isColor = False

    # use the first image as the reference image
    ref_img = images[0]
    if isColor:
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref_img

    # output array
    aligned_images = [ref_img,]

    # find homography between other images and ref img
    for img in images[1:]:
        if isColor:
            # convert it to grayscale for feature extraction
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

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
        if isColor:
            height, width, channels = img.shape
        else:
            height, width = img.shape
        img_warped = cv2.warpPerspective(img, H, (width, height))
        aligned_images.append(img_warped)

    return aligned_images


# Laplacian Pyramid
def get_laplacian_pyramid(img, N):
    """
    returns N-level Laplacian Pyramid of input image as a list
    @input: image
    @output: - Laplacian Pyramid: list of N images containing laplacian pyramids from level 0 to level N
             - Gaussian Pyramid: list of N images containing gaussian pyramids from level 0 to level N
    """
    # current level image
    # curr_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    curr_img = img

    lap_pyramids = []
    gaussian_pyramids = [curr_img,]

    # for N level
    for i in range(N):
        down = cv2.pyrDown(curr_img)
        gaussian_pyramids.append(down)
        up = cv2.pyrUp(down, dstsize=(curr_img.shape[1], curr_img.shape[0]))
        lap = curr_img - up.astype('int16') # NOTE: BE SURE to use int16 instead of cv2.subtract,
                                            #       which cv2 will clip value to 0-255, here we want 
                                            #       arbitratry integeter value.
        lap_pyramids.append(lap)
        curr_img = down
        # top level laplacian be a gaussian downsampled
        if i == N-1:
            lap_pyramids.append(curr_img)

    # # display pyramids images
    # if isPlot:
    #     fg, axs = plt.subplots(2,len(lap_pyramids))
    #     fg.suptitle('Laplacian/Gaussian Pyramid using own method')
    #     for i in range(len(lap_pyramids)):
    #         axs[0, i].imshow(lap_pyramids[i][:,:], cmap='gray')
    #         axs[1, i].imshow(gaussian_pyramids[i][:,:], cmap='gray')
    #     plt.show()

    return lap_pyramids
    

def pyplot_display(images, title='Images Display', gray=False):
    """
    helper function to quickly display a list of images
    @input: - images, the list of images to be displayed
            - title, 'title'
            - gray, is displaying gray scale images?
                otherwise default to display RGB images
    """
    num_img = len(images)
    fg, axs = plt.subplots(1, num_img)
    fg.suptitle(title)
    for i in range(num_img):
        image = images[i]
        if gray:
            axs[i].imshow(image, cmap='gray')
        else:
            if num_img == 1:
                axs.imshow(image[:, :, [2,1,0]])
            else:
                axs[i].imshow(image[:, :, [2,1,0]])
    plt.show()


# Evaluation Metrics:
# - Standard Deviation of Image
def eval_std(image):
    """
    Evaluate image focusness (quality) based on 
        the standard deviation of the image, 
        higher is better.
    @input: image - 2d image (W*H) (grayscale)
    @output: standard deviation score
    """
    mu = np.mean(image)
    sum = 0
    M, N = image.shape
    for m in range(M):
        for n in range(N):
            sum += (image[m, n] - mu)**2 / (M*N)

    return np.sqrt(sum)


# The section below are cited from sweet internet


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





# calculates the E: Entropy for every pixel locations
# Source: https://github.com/sjawhar/focus-stacking/blob/master/focus_stack/pyramid.py - Line 84-105
def get_probabilities(gray_image):
    levels, counts = np.unique(gray_image.astype(np.uint8), return_counts = True)
    probabilities = np.zeros((256,), dtype=np.float64)
    probabilities[levels] = counts.astype(np.float64) / counts.sum()
    return probabilities

def entropy(image, kernel_size):
    def _area_entropy(area, probabilities):
        levels = area.flatten()
        return -1. * (levels * np.log(probabilities[levels])).sum()
    
    probabilities = get_probabilities(image)
    pad_amount = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(image,pad_amount,pad_amount,pad_amount,pad_amount,cv2.BORDER_REFLECT101)
    entropies = np.zeros(image.shape[:2], dtype=np.float64)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(entropies.shape[0]):
        for column in range(entropies.shape[1]):
            area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
            entropies[row, column] = _area_entropy(area, probabilities)

    return entropies


# calculates the D: Deviation for every pixel locations
# Source: https://github.com/sjawhar/focus-stacking/blob/master/focus_stack/pyramid.py - Line 108-122
def deviation(image, kernel_size):
    def _area_deviation(area):
        average = np.average(area).astype(np.float64)
        return np.square(area - average).sum() / area.size

    pad_amount = int((kernel_size - 1) / 2)
    padded_image = cv2.copyMakeBorder(image,pad_amount,pad_amount,pad_amount,pad_amount,cv2.BORDER_REFLECT101)
    deviations = np.zeros(image.shape[:2], dtype=np.float64)
    offset = np.arange(-pad_amount, pad_amount + 1)
    for row in range(deviations.shape[0]):
        for column in range(deviations.shape[1]):
            area = padded_image[row + pad_amount + offset[:, np.newaxis], column + pad_amount + offset]
            deviations[row, column] = _area_deviation(area)

    return deviations


# calculated RE: regional energy for every pixel locations
# Source: https://github.com/sjawhar/focus-stacking/blob/master/focus_stack/pyramid.py - Line 167-169
def region_energy(laplacian):
    return convolve(np.square(laplacian))


def generating_kernel(a):
    kernel = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
    return np.outer(kernel, kernel)


def convolve(image, kernel=generating_kernel(0.4)):
    return cv2.filter2D(src=image.astype(np.float64), ddepth=-1, kernel=np.flip(kernel))