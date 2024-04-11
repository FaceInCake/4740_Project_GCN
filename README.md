# 4740 Project

This is a project for UWindsor's 4740

This project is a Graph based convolutional using super pixels to attempt varing tasks like classification/segmentation/detection/etc...

## Setup

### Python

You need to install [Python](https://www.python.org/downloads/) of at least 3.11

### Jupyter Notebook

Personally, I just use the extension on VScode, and that is what I reccommend

### venv

venv comes with Python, but you'll want to remember to Selec the venv as the Notebooks Kernel and to run any terminal commands from within the environment

### Additional Data!

As you may guess I cant just shove an entire data set into a repo, you'll need:
- The [COCO](https://cocodataset.org/#download) '2017 **Train** Images'
    - Place them into `Data/train2017`
- The [COCO](https://cocodataset.org/#download) '2017 **Val** Images'
    - Place them into `Data/val2017`
- The [COCO](https://cocodataset.org/#download) '2017 **Test** Images'
    - Place them into `Data/test2017`
- A '[Objects_train.csv](https://drive.google.com/file/d/1_5vdmx66YzOWfeyXQ__hLBSS5uIBv__b/view?usp=sharing)' file of parsed annotations, which you can download with the link
    - Place it into ParsedAnnotations
- A '[Segmentations_train.json](https://drive.google.com/file/d/1yQFlueL-TD-b0RTB_pEFpxw_cr4LvDUu/view?usp=sharing)' file of parsed annotations, which you can download with the link
    - Place it into ParsedAnnotations
