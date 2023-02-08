from audioop import avg
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob, os

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
def calc_std(image):
    """
    Evaluate image constrast (quality) based on 
        the standard deviation of the image, 
        higher is better.
    @input: image - 3d image (W*H*C) or 2d
    @output: standard deviation score
    """    
    try:
        sums = []
        M, N, C = image.shape
        for c in range(C):
            mu_c = np.mean(image[:,:,c])
            sum_c = 0
            for m in range(M):
                for n in range(N):
                    sum_c += (image[m, n, c] - mu_c)**2 / (M*N)
            sums.append(sum_c)
        sum = np.mean(sums)

    except:
        mu = np.mean(image)
        sum = 0
        M, N = image.shape
        for m in range(M):
            for n in range(N):
                sum += (image[m, n] - mu)**2 / (M*N)

    return np.sqrt(sum)

def sharpness3d(image):
    """
    calculates sharpness of a 3d image, higher is better
    """
    s = []
    for c in range(image.shape[-1]):
        gy, gx = np.gradient(image[:,:,c])
        gnorm = np.sqrt(gx**2 + gy**2)
        sharpness = np.average(gnorm)
        s.append(sharpness)
    return np.mean(s)

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


def load_images(file_paths):
    """
    Load images in uint8
    """
    images = np.array([cv2.imread(f_name) for f_name in file_paths])

    # check the loaded images are valid
    if any([image is None for image in images]):
        raise RuntimeError("Error loading one or more of the input files.")

    return images

def save_image(image, file_path):
    """
    Save image in RGB format
    """
    cv2.imwrite(file_path, image)
    print(f"Image successfully saved to {file_path}")


def parse_input(input_list):
    """
    @input: an input (list) from argparse,
        if len==1, then the user is specifying a directory
        else, the user is specifying a series of image files
    @output: an list of image file paths
    """
    supported_types = ['*.jpg', '*.jpeg', '*.png'] # the tuple of file types

    file_paths = []
    
    if len(input_list) == 1:
        for ext in supported_types:
            file_paths.extend(glob.glob(os.path.join(input_list[0], ext)))
        if len(file_paths) == 0:
            raise Exception(f"{input_list[0]} does not contain images (.jpg/.jpeg/.png)")
    else:
        # sanity check each file path
        for fp in input_list:
            validate_type = "*." + fp.split(".")[-1]
            if validate_type not in supported_types:
                raise Exception("Input images have to be .jpg/.jpeg/.png")
        file_paths = input_list


    # file num check
    assert len(file_paths) > 1, "Provide at least 2 images."

    return file_paths

def eval(src_images, out_image, verbose=True):
    """
    evaluation of before/after process
    """
    src_std= np.mean([calc_std(image) for image in src_images])
    final_std = calc_std(out_image)

    src_sharp = np.mean([sharpness3d(image) for image in src_images])
    final_sharp = sharpness3d(out_image)

    if verbose:
        print("Evaluate focusness, higher is better:")
        print(f"[Source]: STD DEV = {src_std:.2f}, Sharpness = {src_sharp:.2f}")
        print(f"[Result]: STD DEV = {final_std:.2f}, Sharpness = {final_sharp:.2f}")
        print("NOTE: STDDEV represents constrast and might be depricated.")

    return src_std, final_std, src_sharp, final_sharp

def plot(src_images, out_image):
    """
    plot final results side by side comparing with source images (BGR image)
    """
    # convert to uint8
    out_image = cv2.normalize(out_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    # convert to RGB just for plotting
    src_images = BGR2RGBs(src_images)
    out_image = BGR2RGB(out_image)
    
    #subplot(r,c) provide the no. of rows and columns
    fig, ax = plt.subplots(1, len(src_images)+1, figsize=(10, 7)) 
    fig.tight_layout(pad=1.0)

    for i, src_img in enumerate(src_images):
        ax[i].imshow(src_img)
        ax[i].set_title(f"Source Image {i+1}")
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    ax[-1].imshow(out_image)
    ax[-1].set_title("Final Result")
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])

    src_std, final_std, src_sharp, final_sharp = eval(src_images, out_image)

    plt.figtext(0.99, 0.01, f"Source Sharpness= {src_sharp:.2f} | Result Sharpness = {final_sharp:.2f}", horizontalalignment='right')
    plt.show()


# color conversions
def BGR2RGBs(images):
    return np.array([BGR2RGB(img) for img in images])

def BGR2RGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def RGB2BGRs(images):
    return np.array([RGB2BGR(img) for img in images])

def RGB2BGR(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
