# Setup an AWS GPU Cluster from Stratch

In this documents we give a step-by-step tutorial on how to setup Amazon AWS for
MXNet. In particular, we will address:

- [Use Amazon S3 to host data](#use-amazon-s3-to-host-data)
- [Setup EC2 GPU instance with all dependencies installed](#setup-an-ec2-gpu-instance)
- [Build and Run MXNet on a single machine](#build-and-run-mxnet-on-a-gpu-instance)
- [Setup an EC2 GPU cluster for distributed training](#setup-an-ec2-gpu-cluster)

## Use Amazon S3 to host data

Amazon S3 is distributed data storage, which is quite convenient for host large
scale datasets. In order to S3, we need first to get the
[AWS credentials](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html)),
which includes a `ACCESS_KEY_ID` and a `SECRET_ACCESS_KEY`.

In order for MXNet to use S3, we only need to set the environment variables `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` properly. For example, we can add the following two lines in
`~/.bashrc` (replace the strings with the correct ones)

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

There are several ways to upload local data to S3. One simple way is using
[s3cmd](http://s3tools.org/s3cmd). For example:

```bash
wget http://webdocs.cs.ualberta.ca/~bx3/data/mnist.zip
unzip mnist.zip && s3cmd put t*-ubyte s3://dmlc/mnist/
```

## Setup an EC2 GPU Instance

MXNet requires the following libraries

- C++ compiler with C++11 suports, such as `gcc >= 4.8`
- `CUDA` (`CUDNN` in optional) for GPU linear algebra
- `BLAS` (cblas, open-blas, atblas, mkl, or others) for CPU linear algebra
- `opencv` for image augmentations
- `curl` and `openssl` for read/write Amazon S3

Installing `CUDA` on EC2 instances needs a little bit effects. Caffe has a nice
[tutorial](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN))
on how to install CUDA 7.0 on Ubuntu 14.04 (Note: we tried CUDA 7.5 on Nov 7
2015, but it is problematic.)

The reset can be installed by the package manager. For example, on Ubuntu:

```
sudo apt-get update
sudo apt-get install -y build-essential git libcurl4-openssl-dev libatlas-base-dev libopencv-dev python-numpy
```

We provide a public Amazon Machine Images, [ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178), with the above packages installed.


## Build and Run MXNet on a GPU Instance

The following commands build MXNet with CUDA/CUDNN, S3, and distributed
training.

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
echo "USE_BLAS=atlas" >> config.mk
echo "USE_DIST_KVSTORE = 1" >>config.mk
echo "USE_S3=1" >>config.mk
make -j8
```

Test if every goes well, we train a convolution neural network on MNIST using GPU:

```bash
python tests/python/gpu/test_conv.py
```

If the MNISt data is placed on `s3://dmlc/mnist`, we can let the program read
the S3 data directly:

```bash
sed -i.bak "s!data_dir = 'data'!data_dir = 's3://dmlc/mnist'!" tests/python/gpu/test_conv.py
```

Note: We can use `sudo ln /dev/null /dev/raw1394` to fix the opencv error `libdc1394 error: Failed to initialize libdc1394`.

## Setup an EC2 GPU Cluster

A cluster consists of multiple machines. We can use the machine with MXNet
installed as the root machine for submitting jobs, and then launch several
slaves machine to run the jobs. For example, launch multiple instances using a
AMI, e.g.
[ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178),
with dependencies installed. There are two suggestions:

1. Make all slaves' ports are accessible (same for the root) by setting **type: All TCP**,
   **Source: Anywhere** in **Configure Security Group**

2. Use the same `pem` as the root machine to access all slaves machines. And
   then cpy the `pem` file into root machine's `~/.ssh/id_rsa`, it all slaves
   machines are ssh-able from the root.

Now we run the previous CNN on multiple machines. Assume we are on a working
directory of the root machine, such as `~/train`, and MXNet is built as `~/mxnet`.

1. First pack the mxnet python library into this working directory for easy
  synchronization:

  ```bash
  cp -r ~/mxnet/python/mxnet .
  cp ~/mxnet/lib/libmxnet.so mxnet/
  ```

  And then copy the training program:

  ```bash
  cp ~/mxnet/example/distributed-training/*mnist* .
  ```

2. Prepare a host file with all slaves's private IPs. For example, `cat hosts`

  ```bash
  172.30.0.172
  172.30.0.171
  ```

3. Assume there are 10 slaves, then train the CNN using 10 workers and 10 servers:

  ```bash
  ~/mxnet/tracker/dmlc_ssh.sh -n 10 -s 10 -H hosts python train_mnist.py
  ```

Here we use a simple ```dmlc_ssh``` that runs DMLC jobs without any cluster frameworks.

Note: Sometimes the jobs lingers at the slave machines even we pressed `Ctrl-c`
at the root node. We can kill them by

```bash
cat hosts | xargs -I{} ssh -o StrictHostKeyChecking=no {} 'uname -a; pgrep python | xargs kill -9'
```

Note: The above example is quite simple to train and therefore is not a good
benchmark for the distributed training. We may consider other examples such as
[imagenet using inception network](https://github.com/dmlc/mxnet/tree/master/example/distributed-training/train_imagenet.py)

## More NOTE
### Use multiple data shards
Usually it is common to pack dataset into multiple files, especially when we pack it distributedly. MXNet support direct loading from multiple data shards, simply put all the record files into a folder, and point the data path to the folder

### Use YARN, MPI, SGE
While ssh can be simple for cases when we do not have a cluster scheduling framework. MXNet is designed to be able to port to various platforms.  We also provide other scripts in [tracker](https://github.com/dmlc/dmlc-core/tree/master/tracker) to run on other cluster frameworks, including Hadoop(YARN) and SGE. Your contribution is more than welcomed to provide examples to run mxnet on your favorite distributed platform.
  

