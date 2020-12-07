import numpy as np
import cv2
import matplotlib.pyplot as plt

import os, sys, glob, argparse

from helper import align_images, get_laplacian_pyramid, entropy, deviation, region_energy, pyplot_display

intro = \
"""
Focus Stacking

This project is the final project of CS445 Fall2020,
Team: Zongnan Bao(zb3) and Han Chen(hanc3)\n\n
"""

simple = \
"""
stack photos with different depth of fields
"""

# focus_stacking (naive method, depricated)
def naive_focus_stacking(images):
    """
    achieves the functionality of focus stacking using max of LoG masking (pixel-wise)
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

    # # TODO: crop image so that it fits in the largest rectangle in warped image
    # for i in range(len(aligned_gray)):


    # 2 - Gaussian blur on all images
    for i in range(len(aligned_gray)):
        aligned_gray[i] = cv2.GaussianBlur(aligned_gray[i], ksize=(3, 3), sigmaX=3)

    # 3 - Laplacian on blurred images
    for i in range(len(aligned_gray)):
        aligned_gray[i] = cv2.Laplacian(aligned_gray[i], cv2.CV_64F, ksize=3)

    # display LoG
    fg, axs = plt.subplots(1,3)
    fg.suptitle('Laplacian of Gaussian(edge detect) for all images')
    axs[0].imshow(np.absolute(aligned_gray[0][:,:]))
    axs[1].imshow(np.absolute(aligned_gray[1][:,:]))
    axs[2].imshow(np.absolute(aligned_gray[2][:,:]))
    plt.show()

    # prepare output image
    canvas = np.zeros(images[0].shape)

    # get the maxmimum value of LoG across all images
    max_LoG = np.max(np.absolute(aligned_gray), axis=0)

    # find mask corresponding to maximum (which LoG ahiceves the maximum)
    masks = (np.absolute(aligned_gray) == max_LoG).astype('uint8')

    # display masks
    fg, axs = plt.subplots(1,3)
    fg.suptitle('MASKs for all images')
    axs[0].imshow(masks[0])
    axs[1].imshow(masks[1])
    axs[2].imshow(masks[2])
    plt.show()
    
    # apply masks
    for i in range(len(aligned_images)):
        canvas = cv2.bitwise_not(aligned_images[i], canvas, mask=masks[i])

    # process the output
    canvas = (255-canvas)
    return canvas



# focus-stacking (laplacian pyramid fusion method)
def lap_focus_stacking(images, N=5, kernel_size=5):
    """
    achieves the functionality of focus stacking using Laplacian Pyramid Fusion described 
        in Wang and Chang's 2011 paper (regional fusion)
    @input: images - array of images
            N      - Depth of Laplacian Pyramid
            kernel_size - integer represents the side of Gaussian kernel
    @output: single image that stacked the depth of fields of all images
    """
    # # 1 - align images
    # images = align_images(images)

    # 2- generate array of laplacian pyramids
    list_lap_pyramids = []
    for img in images:
        # get both laplacian and gaussian pyramid
        lap_pyr = get_laplacian_pyramid(img, N)
        base = lap_pyr[-1]
        lap_pyr = lap_pyr[:-1]
        list_lap_pyramids.append(lap_pyr)
    list_lap_pyramids = np.array(list_lap_pyramids, dtype=object)

    LP_f = []

    # 3 - regional fusion using these laplacian pyramids
    # fuse level = N laplacian pyramid
    D_N = np.array([deviation(lap, kernel_size) for lap in list_lap_pyramids[:, -1]])
    E_N = np.array([entropy(lap, kernel_size) for lap in list_lap_pyramids[:, -1]])

    # 3.1 - init level N fusion canvas
    LP_N = np.zeros(list_lap_pyramids[0, -1].shape)
    for m in range(LP_N.shape[0]):
        for n in range(LP_N.shape[1]):
            D_max_idx = np.argmax(D_N[:, m, n])
            E_max_idx = np.argmax(E_N[:, m, n])
            D_min_idx = np.argmin(D_N[:, m, n])
            E_min_idx = np.argmin(E_N[:, m, n])
            # if the image maximizes BOTH the deviation and entropy, use the pixel from that image
            if D_max_idx == E_max_idx:
                LP_N[m, n] = list_lap_pyramids[D_max_idx, -1][m, n]
            # if the image minimizes BOTH the deviation and entropy, use the pixel from that image
            elif D_min_idx == E_min_idx: 
                LP_N[m, n] = list_lap_pyramids[D_min_idx, -1][m, n]
            # else average across all images
            else:
                for k in range(list_lap_pyramids.shape[0]):
                    LP_N[m, n] += list_lap_pyramids[k, -1][m, n]
                LP_N[m, n] /= list_lap_pyramids.shape[0]

    LP_f.append(LP_N)

    # 3.2 - fusion other levels of laplacian pyramid (N-1 to 0)
    for l in reversed(range(0, N-1)):
        # level l final laplacian canvas
        LP_l = np.zeros(list_lap_pyramids[0, l].shape)

        # region energey map for level l
        RE_l = np.array([region_energy(lap) for lap in list_lap_pyramids[:, l]], dtype=object)

        for m in range(LP_l.shape[0]):
            for n in range(LP_l.shape[1]):
                RE_max_idx = np.argmax(RE_l[:, m, n])
                LP_l[m, n] = list_lap_pyramids[RE_max_idx, l][m, n]

        LP_f.append(LP_l)

    LP_f = np.array(LP_f, dtype=object)
    LP_f = np.flip(LP_f)

    # display fused pyramids images
    if isPlot: 
        pyplot_display(LP_f, title='Fused Laplacian Pyramid Map', gray=True)

    # 4 - time to reconstruct final laplacian pyramid(LP_f) back to original image!
    # get the top-level of the gaussian pyramid
    fused_img = cv2.pyrUp(base, dstsize=(LP_f[-1].shape[1], LP_f[-1].shape[0])).astype(np.float64)

    for i in reversed(range(N)):
        # combine with laplacian pyramid at the level
        fused_img += LP_f[i]
        if i != 0:
            fused_img = cv2.pyrUp(fused_img, dstsize=(LP_f[i-1].shape[1], LP_f[i-1].shape[0]))
    
    return fused_img



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=simple)

    # parse path to input folder
    parser.add_argument('input_path', type=str, help='path to the directory containing input images')
    parser.add_argument('--output_name', type=str, default='output.jpg',help='the output file name, default will be \'output.jpg\'')
    parser.add_argument('--plot', action='store_true', help='run with this flag to show all process using matplotlib.')
    parser.add_argument('--depth', type=int, default=5, help='depth(level) of Laplacian Pyramid, default to 5')
    parser.add_argument('--k_size', type=int, default=5, help='kernel size of Gaussian Blurring used in pyramid')

    args = parser.parse_args()

    # extract args
    dir_path = args.input_path
    output_name = args.output_name
    global isPlot
    isPlot = args.plot
    pyramid_depth = args.depth
    kernel_size = args.k_size
    
    # load images
    file_names = [img for img in glob.glob(os.path.join(dir_path, '*.jpg'))]
    
    num_files = len(file_names)
    
    # input sanity checks
    assert num_files > 1, "Provide at least 2 images."

    # load images (in HSV)
    images = np.array([cv2.cvtColor(cv2.imread(f_name), cv2.COLOR_BGR2HSV) for f_name in file_names])
    V_channel = 2
    
    # check the filenames are valid
    if any([image is None for image in images]):
        raise RuntimeError("Cannot load one or more input files.")

    # display original images
    if isPlot:
        pyplot_display(images[:, :, :, V_channel], title='Unprocessed Images (grayscale)', gray=True)

    # focus stacking
    canvas = lap_focus_stacking(images[:, :, :, V_channel], N=pyramid_depth, kernel_size=kernel_size)
    images[0][:,:,V_channel] = canvas

    # show gray result
    if isPlot:
        pyplot_display(canvas, title='Final Result (grayscale)', gray=True)
        pyplot_display(images[0], title='Colored Final Result (work in progress...)', gray=False)

    # write to file (grayscale)
    cv2.imwrite(output_name, canvas)