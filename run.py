import argparse

from src.utils import load_images, save_image, parse_input, eval_std
from src.stack import naive_focus_stacking, lap_focus_stacking_3d

simple = \
"""
stack photos with different depth of fields
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=simple)

    # parse path to input folder
    parser.add_argument('input_path', type=str, nargs='+', help='path to the directory or paths to multiple images')
    parser.add_argument('-o', '--output_path', type=str, default='./result.jpg',help='the output file path, default will be \'./result.jpg\'')
    parser.add_argument('-d', '--depth', type=int, default=5, help='depth(level) of Laplacian Pyramid, default to 5')
    parser.add_argument('-k', '--k_size', type=int, default=5, help='kernel size of Gaussian Blurring used in pyramid')
    parser.add_argument('--plot', action='store_true', help='run with this flag to plot results')
    parser.add_argument('--debug', action='store_true', help='run with this flag to show all debugging process.')
    parser.add_argument('--naive', action='store_true', help='run with this flag to use naive method (max LoG)')
    parser.add_argument('--eval', action='store_true', help='run with this flag to evaluate the focusness before/after focus stacking using standard deviation')

    args = parser.parse_args()

    # parse input paths
    file_paths = parse_input(args.input_path)

    # load images in HSV
    images = load_images(file_paths)

    # focus stack
    if args.naive:
        canvas = naive_focus_stacking(images, debug=args.debug)
    else:
        # laplacian pyramid method
        canvas = lap_focus_stacking_3d(images, N=args.depth, kernel_size=args.k_size, debug=args.debug)

    # save file
    save_image(canvas, args.output_path)

    # evaluation
    if args.eval:
        print("Evaluate focusness using std dev, higher is better:")
        src_std= max([eval_std(image) for image in images])
        print("- MAX Std dev before focus stacking: {}".format(src_std))

        final_std = eval_std(canvas)
        print("- Std dev after focus stacking: {}".format(final_std))