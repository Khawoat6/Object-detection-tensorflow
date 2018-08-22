Object Detection Using TensorFlow 
========================
This repository is a tutorial for how to use TensorFlow's Object Detection API<br>

## Steps
#### 1. Install NVIDIA cuDNN, CUDA Toolkit and Reboot computer
https://developer.nvidia.com/cuda-90-download-archive (CUDA Toolkit 9.0)<br>
https://developer.nvidia.com/rdp/cudnn-download (cuDNN v7.0.5 (Dec 5, 2017), for CUDA 9.0)

#### 2. Install Anaconda
https://www.anaconda.com/download/

#### 3. Install TensorFlow-GPU
In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:
```
C:\> conda create -n tensorflow1 pip python=3.5
```
Then, activate the environment by issuing:
```
C:\> activate tensorflow1
```
Install tensorflow-gpu in this environment by issuing:
```
(tensorflow1) C:\> pip install --ignore-installed --upgrade tensorflow-gpu
```
Install the other necessary packages by issuing the following commands:
```
(tensorflow1) C:\> pip install pillow
(tensorflow1) C:\> pip install Cython
(tensorflow1) C:\> pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
(tensorflow1) C:\> pip install pandas
(tensorflow1) C:\> pip install lxml
(tensorflow1) C:\> pip install contextlib2
(tensorflow1) C:\> pip install opencv-python
```
https://github.com/philferriere/cocoapi to install pycocotools

#### 4. Configure PYTHONPATH environment variable
You can choose one method to create PYTHONPATH

```
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research\slim
```

Or in Search, search for and then select: <b>System (Control Panel)</b><br>
Click the <b>System</b> and then <b>Advanced system settings</b> link.<br>
Click <b>Environment Variables</b>. In the section System Variables, click <b>new</b>.<br>
<b>Name:</b> PYTHONPATH<br>
<b>Value:</b> C:\tensorflow1\models;C:\tensorflow1\models\research\slim<br>
<b>Click OK</b>. Close all remaining windows by clicking OK.

#### 5. Compile Protobufs and run setup.py
You can download it here <a href="https://github.com/google/protobuf/releases/tag/v3.0.2">Protocol Buffers v3.0.2</a>
```
(tensorflow1) C:\> cd C:\tensorflow1\models\research
(tensorflow1) C:\tensorflow1\models\research> C:/Users/name/Desktop/bin/protoc object_detection/protos/*.proto --python_out=.
```
Finally, run the following commands from the C:\tensorflow1\models\research directory:
```
(tensorflow1) C:\tensorflow1\models\research> python setup.py build
(tensorflow1) C:\tensorflow1\models\research> python setup.py install
```

<br>

## Trained model
We'll use faster_rcnn_inception_v2_coco models.

We can download models here [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)

Once downloaded, unzip the folder in /object_detection. The folder must have file frozen_inference_graph.pb (models)

<br>

## Retraining model
We use for example faster_rcnn_inception_v2_coco models

Save folder faster_rcnn_inception_v2_coco_2018_01_28 in C:\tensorflow1\models\research\object_detection

#### 1. Create Label Map and Configure Training
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:\tensorflow1\models\research\object_detection\output folder, will look like. 
```
item {
  id: 1
  name: 'raccoon'
}
```

Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training.

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the faster_rcnn_inception_v2_pets.config file into the \object_detection\output directory and change "PATH_TO_BE_CONFIGURED",num_classe,num_examples for correct data, like <a href="https://github.com/JMonda/Object-detection-tensorflow/blob/master/output/faster_rcnn_inception_v2_pets.config">here</a>

#### 2. Generate Training Data
We have to save our image in imagenes/

First, the image .xml data will be used to create TFRecord files containing all the data for the images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python create_pet_tf_record.py --data_dir=imagenes --output_dir=output --label_map_path=output/labelmap.pbtxt
```

#### 3. Run the Training
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python model_main.py --pipeline_config_path=output/faster_rcnn_inception_v2_pets.config --model_dir=trained_model --alsologtostderr
```

If you want to see all the graphs, run this command
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> tensorboard --logdir=trained_model
```
<img src="https://i.imgur.com/3GTcNdL.png" />
<img src="https://i.imgur.com/gq6ph8y.png" />

#### 4. Export Models
Now that training is complete, the last step is to generate the frozen inference graph (.pb file).

“XXX” in “model.ckpt-XXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python export_inference_graph.py --pipeline_config_path=output/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix=trained_model/model.ckpt-XXX --output_directory=inference_graph
```
the file frozen_inference_graph.pb has been generated, this is your models

<br>

## How to run the models
Now you have your models (frozen_inference_graph.pb).

<br>To run it
```
(tensorflow1) C:\tensorflow1\models\research\object_detection> python Object_detection_webcam.py
```

Output<br>
<img src="https://i.imgur.com/O6fRy4l.png" />
