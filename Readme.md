# Faster RCNN

This is the Keras implementation of Faster RCNN architecture: Towards Real Time Object Detection with Region Proposal Network (RPN)

### **Note:**

- Only tensorflow backend is supported and this requires tensorflow version 1.x
- Pydot and Graphviz packages need to be installed to visualize model architecture

## Features:

- Tensorboard support for logging metrics and visualizing model graph
- Easy implementation for other feature extractors like VGG
- Terminal support for quick and easy execution

## Usage:

### **Training:**
- [train_frcnn](train_frcnn.py) is used to train the model

```python
        python train_frcnn.py -p sample.txt
```
- The input data should be a text file of training set with each line in the following format

  `filepath,x1,y1,x2,y2,classname`

  ### Example:
  `/path/image_1.jpg,214,495,316,618,person`

  `/path/image_2.jpg,487,338,632,411,dog`

- The model outputs the following:
   - Config pickle file – used during testing
   - Model weights in hdf5 format 
   - A CSV file containing all the training metrics
   - Full model image for better understanding of the model architecture
   - Event file for visualizing model graph in tensorboard

**Note:**
- Input image size is set to 400. This can be changed in [config.py](keras_frcnn/config.py).
- Pre-trained weights for ResNet can be downloaded from [here](https://github.com/fchollet/deep-learning-models/releases/tag/v0.2)


### **Testing:**
- [test_frcnn](test_frcnn.py) is used to test the model
```python
        python test_frcnn.py -p test_images/ - a images.txt
```
- The input arguments should be a path to the test images folder and a text file with each line in the following format. The text file is needed to calculate mAP.

  `filepath,x1,y1,x2,y2,classname`

  ### Example:
  `/path/image_1.jpg,214,495,316,618,person`

  `/path/image_2.jpg,487,338,632,411,dog`

- The test file outputs the following:
   -	Text files for each image with predicted bounding box coordinates, confidence score and its class (saved to a folder)
   -	Test images with bounding box predictions (saved to a folder)
   -	Prints the Mean Average Precision score
   -	An image of Precison-Recall curve


**Note:**
- The index slicing should grab only the filename and not the file path in [test_frcnn](test_frcnn.py) file. This is important for calculating mAP. 
```python
        with open(mean_ap, 'r') as f:
		for line in f:
			line_split = line.strip().split(',')
			(filename, x1, y1, x2, y2, classname) = line_split
			filename1 = filename[7:]
			if img_name == filename1:
				w = int(x2)-int(x1)
				h = int(y2)-int(y1)
				gt.append([str(filename1), int(x1), int(y1), w, h])
			else:
				pass
```

- Change the input image size in [mean_avg](mean_avg.py) file before testing
```python
        gtFormat = BBFormat.XYWH
        detFormat = BBFormat.XYWH
        gtCoordType = CoordinatesType.Absolute
        detCoordType = CoordinatesType.Absolute
        imgSize = (800, 800)
```

## API Reference: 
https://github.com/tensorflow/models/tree/master/research/object_detection

## References 
- Implementation adopted from [here](https://github.com/yhenon/keras-frcnn/) and is based on the idea of tensorflow’s object detection [API](https://github.com/tensorflow/models/tree/master/research/object_detection).
-  Mean Average Precision implementation is from [here](https://github.com/rafaelpadilla/Object-Detection-Metrics)