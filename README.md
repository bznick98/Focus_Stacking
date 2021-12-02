![](https://visitor-badge.glitch.me/badge?page_id=bznick98.Focus_Stacking)
# Focus Stacking

This project is the final project of CS445 Fall2020,
Team: Zongnan Bao(zb3) and Han Chen(hanc3). It's guided by Wang and Chang's 2011 paper: ***[A Multi-focus Image Fusion Method Based on Laplacian Pyramid](https://pdfs.semanticscholar.org/9079/27b96fa87283efbc5f9a9a4202a7f8e879ff.pdf?_ga=2.245832256.531696024.1607257913-188213619.1607257913)***, and some other internet resources.
This tool has two method implemented for blending images with different depth of field, we recommend using the default one (Laplacian Pyramid Method) since it produces better results.

## Demo:
![Demo Tie](./demo.png)

## Workflow:
![Image of project workflow](./workflow.png)
[> Detailed Report](Final%20Project%20Docs/CS445%20Final%20Project%20Report.pdf)

## How To Use:
Clone this repo and run ***focus_stack.py*** with path to a DIRECTORY that contains input image files.
```console
python3 focus_stack.py path_to_directory [optional: arguments and flags]
```
**like this**, 
```console
python3 focus_stack.py test_img/test6
```
or with arguments,

```console 
python3 focus_stack.py test_img/test6 --depth 10
```
The final result will be stored in `output.jpg` or the name you specified.

## Some optional arguments and flags:
#### Arguments:
* --depth [INT] : the depth (level) of Laplacian Pyramid, default = 5.
* --k_size [INT] : the kernel size of Gaussian Kernel (an integer), default = 5.
* --output_name [STR] : the name of output file, default = 'output.jpg'.

#### Flags:
* -h : show help info, all descriptions of args and flags.
* -plot : plot laplacian pyramids, original images, final result step by step.
* -naive : to use naive method (max LoG) for focus stacking.
* -eval : to evaluate the focusness(quality) before/after focus stacking using standard deviation, higher is better.

## Language and Dependent Packages:
* Python3
* OpenCV
* Matplotlib

## Source:
* Wang & Chang 2011, ***[A Multi-focus Image Fusion Method Based on Laplacian Pyramid](https://pdfs.semanticscholar.org/9079/27b96fa87283efbc5f9a9a4202a7f8e879ff.pdf?_ga=2.245832256.531696024.1607257913-188213619.1607257913)***
* [Test6 folder images source](https://www.cambridgeincolour.com/tutorials/focus-stacking.htm)
* Some code snippets cited inside helper.py

## TODO:
* Support colored image for laplacian method. (Done!)
* Fix align image error for laplacian method, there is no align step right now. 
* Add GUI to the tool, make it easy to use for non-tech photography community.
