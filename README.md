## PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

### Installation
**Requirements**
  - TensorFlow
  - python
  - numpy

### Compile Customized TF Operators
The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.
 1. Update `nvcc` and `python` path if necessary. 
 2. Then, find Tensorflow include and library paths.
    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
    add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands. 
 3. Finally, compile by type `./tf_xx_compile.sh` under the corresponding folder.


Note:
 1. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

### Usage

#### Shape Classification

To train a PointNet++ model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

        python train.py

To see all optional arguments for training:

        python train.py -h

If you have multiple GPUs on your machine, you can also run the multi-GPU version training (our implementation is similar to the tensorflow <a href="https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10">cifar10 tutorial</a>):

        CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2

After training, to evaluate the classification accuracies (with optional multi-angle voting):

        python evaluate.py --num_votes 12 

<i>Side Note:</i> For the XYZ+normal experiment reported in our paper: (1) 5000 points are used and (2) a further random data dropout augmentation is used during training (see commented line after `augment_batch_data` in `train.py` and (3) the model architecture is updated such that the `nsample=128` in the first two set abstraction levels, which is suited for the larger point density in 5000-point samplings.

To use normal features for classification: You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. Move the uncompressed data folder to `data/modelnet40_normal_resampled`

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

        cd part_seg
        python train.py

Preprocessed ShapeNetPart dataset (XYZ, normal and part labels) can be found <a href="https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip">here (674MB)</a>. Move the uncompressed data folder to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

#### Semantic Scene Parsing

See `scannet/README` and `scannet/train.py` for details.

#### Visualization Tools
We have provided a handy point cloud visualization tool under `utils`. Run `sh compile_render_balls_so.sh` to compile it and then you can try the demo with `python show3d_balls.py` The original code is from <a href="http://github.com/fanhqme/PointSetGeneration">here</a>.

#### Prepare Your Own Data
You can refer to <a href="https://github.com/charlesq34/3dmodel_feature/blob/master/io/write_hdf5.py">here</a> on how to prepare your own HDF5 files for either classification or segmentation. Or you can refer to `modelnet_dataset.py` on how to read raw data files and prepare mini-batches from them. A more advanced way is to use TensorFlow's dataset APIs, for which you can find more documentations <a href="https://www.tensorflow.org/programmers_guide/datasets">here</a>.

