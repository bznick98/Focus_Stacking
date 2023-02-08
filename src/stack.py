import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.utils import align_images, get_laplacian_pyramid, entropy, deviation, region_energy, pyplot_display


def naive_focus_stacking(images, debug=False):
    """
    achieves the functionality of focus stacking using max of LoG masking (pixel-wise)
    @input: array of images
    @output: single image that stacked the depth of fields of all images
    """
    # determine input image is gray or not
    isColor = True
    if len(images.shape) <= 3:
         isColor = False

    # 1 - align images
    aligned_images = align_images(images)

    if debug:
        # display alignment
        pyplot_display([images[0], aligned_images[1]], title='Reference Image and the Second Image(Aligned)', gray=(not isColor))
    
    if isColor:
        # convert to grayscale to get largest rectangle & Laplacian of Gaussian
        aligned_gray = [cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY) for aligned_img in aligned_images]
    else:
        aligned_gray = aligned_images

    # convert to binary map to perform max-size rectangle
    aligned_bin, thre = cv2.threshold(aligned_gray[1], 0, 1, cv2.THRESH_BINARY)
    if debug:
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

    if debug:
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

    if debug:
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


def lap_focus_stacking_3d(images, N=5, kernel_size=5, debug=False):
    lap_fs_layers = []
    
    # fuse each channel separately
    for c in range(3):
        lap_fs_layers.append(lap_focus_stacking(images[:, :, :, c], N=N, kernel_size=kernel_size, debug=debug))
    
    lap_fs_layers = np.array(lap_fs_layers)
    # CxWxH -> WxHxC
    lap_fs_layers = np.moveaxis(lap_fs_layers, 0, -1)
    return lap_fs_layers

def lap_focus_stacking(images, N=5, kernel_size=5, debug=False):
    """
    achieves the functionality of focus stacking using Laplacian Pyramid Fusion described 
        in Wang and Chang's 2011 paper (regional fusion)
    @input: images - array of images
            N      - Depth of Laplacian Pyramid
            kernel_size - integer represents the side length of Gaussian kernel
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
    # fuse level = N laplacian pyramid, D=deviation, E=entropy
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
    if debug: 
        pyplot_display(LP_f, title='Fused Laplacian Pyramid Map', gray=True)

    # 4 - time to reconstruct final laplacian pyramid(LP_f) back to original image!
    # get the top-level of the gaussian pyramid
    fused_img = cv2.pyrUp(base, dstsize=(LP_f[-1].shape[1], LP_f[-1].shape[0])).astype(np.float64)

    for i in reversed(range(N)):
        # combine with laplacian pyramid at the level
        fused_img += LP_f[i]
        if i != 0:
            fused_img = cv2.pyrUp(fused_img, dstsize=(LP_f[i-1].shape[1], LP_f[i-1].shape[0]))
    
    # in float64
    return fused_img