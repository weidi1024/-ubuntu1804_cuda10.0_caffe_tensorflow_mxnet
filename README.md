### 实验室电脑　安装ubuntu18.04 caffe tensorflow mxnet　过程

#### 电脑硬件
DELL-PRECSION TOWER 7910 <br> 
Intel® Xeon(R) CPU E5-2630 v4 @ 2.20GHz × 20  <br> 
GeForce GTX 1080 Ti/PCIe/SSE2 <br> 
内存125.8 GiB <br> 

------------

#### 安装系统
下载ubuntu1804系统安装包

#### apt换源
打开软件和更新
ubuntu软件-下载自-进入修改为国内的地址
关闭并更新
升级
sudo apt-get update
sudo apt-get upgrade

#### 安装pip
`
sudo apt-get install python-pip python-dev build-essential
sudo pip install --upgrade pip
`
### pip换源
 修改 ~/.pip/pip.conf (没有就创建一个)， 如下：
`mkdir ~/.pip/
geidt ~/.pip/pip.conf
`
输入如下内容并保存
```
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
```
pip换源之后速度真的是非常舒爽...




### 安装显卡驱动
430.34
安装依赖项
```bash
sudo apt-get install dkms build-essential linux-headers-generic apt-show-versions
```
禁用nouveau:
打开文件
```bash
sudo gedit /etc/modprobe.d/blacklist.conf

```
在最后一行加入
```bash
blacklist nouveau
blacklist lbm-nouveau
options nouveau modeset=0
alias nouveau off
alias lbm-nouveau off
```
保存，并执行如下命令
```bash
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
```
重启后开始安装驱动
```bash
sudo ./NVIDIA-Linux-*.run
```
终端执行
```bash
nvidia-smi
```
出现如下信息就说明安装成功了


【安装相关依赖项】
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libboost-all-dev libopenblas-dev liblapack-dev libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev


【安装cuda10.0】
sudo ./runpackage/cuda_9.0.176_384.81_linux.run
sudo gedit ~/.bashrc
写入
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
设置环境变量和动态链接库
sudo gedit /etc/profile
写入export PATH=/usr/local/cuda/bin:$PATH
sudo gedit /etc/ld.so.conf.d/cuda.conf
写入/usr/local/cuda/lib64
sudo ldconfig
reboot重启
nvcc --version
测试一下看看cuda是否安装成功
cd /home/weidi/NVIDIA_CUDA-9.0_Samples/1_Utilities/deviceQuery 
sudo make -j4
./deviceQuery
最后一行PASS即可

【安装cudnn7.4】
解压
sudo cp include/cudnn.h /usr/local/cuda/include
sudo cp lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
对，直接拷贝过去就可以了`。`


【安装相关依赖项】
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libtiff5-dev libdc1394-22-dev libatlas-base-dev gfortran



【安装openblas】
sudo apt-get install libopenblas-dev
sudo apt-get install libopenblas-base
 可以到 https://github.com/xianyi/OpenBLAS/releases 下载你喜欢的版本解压到指定目录，也可以直接git clone
cd OpenBLAS
make -j10
sudo make --PREFIX=/usr/local/OpenBLAS/ install
测试OpenBLAS
gcc ./lib/test.c  -I /usr/local/OpenBLAS/include/ -L /usr/local/OpenBLAS/lib -lopenblas
生成可执行文件　a.out
a.out
结果如下
0.000000 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000 8.000000 9.000000 
90.000000 81.000000 72.000000 63.000000 54.000000 45.000000 36.000000 27.000000 18.000000 9.000000


【安装caffe-ssd】
修改makeconfig，文件就在当前目录下，直接将当前目录下的拷贝到caffe-ssd中即可
查看opencv版本:pkg-config opencv --modversion


编译
sudo make all -j20
sudo make test -j20
sudo make runtest -j20
sudo make pycaffe -j20
	【错误１】
collect2: error: ld returned 1 exit status
.build_release/lib/libcaffe.so：对‘cv::VideoWriter::isOpened() const’未定义的引用
collect2: error: ld returned 1 exit status
Makefile:619: recipe for target '.build_release/tools/convert_annoset.bin' failed
make: *** [.build_release/tools/convert_annoset.bin] Error 1
Makefile:624: recipe for target '.build_release/examples/siamese/convert_mnist_siamese_data.bin' failed
make: *** [.build_release/examples/siamese/convert_mnist_siamese_data.bin] Error 1
	【解决１】
首先这个问题确实是opencv的问题，只需要把  Makefile.config里的#USE_PKG_CONFIG := 这一行前面的#给去掉，然后在他下一行添加
LIBRARIES += glog gflags protobuf leveldb snappy \
        lmdb boost_system hdf5_hl hdf5 m \
        opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs
原文：https://blog.csdn.net/yuweiyang123/article/details/53106638 
	【警告２】
In file included from src/caffe/util/math_functions.cu:1:0:
/usr/local/cuda/include/math_functions.h:54:2: warning: #warning "math_functions.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead." [-Wcpp]
 #warning "math_functions.h is an internal header file and must not be used directly.  This file will be removed in a future CUDA release.  Please use cuda_runtime_api.h or cuda_runtime.h instead."
	【解决２】
修改头文件即可


【错误３】
1 Check failed: a <= b <0 vs -1.19209e-007>
【解决３】
https://blog.csdn.net/LuohenYJ/article/details/88416180
解决办法修改src/caffe/util/sampler.cpp，如下面修改代码所示//renew注释下，加入两个判断，使得bbox长宽不要越界。
  // Figure out bbox dimension.
  float bbox_width = scale * sqrt(aspect_ratio);
  float bbox_height = scale / sqrt(aspect_ratio);
 
  //renew
  if(bbox_width>=1.0)
  {
    bbox_width=1.0;
  }
  if(bbox_height>=1.0)
  {
    bbox_height=1.0;
  }




【错误４】
./include/caffe/util/cudnn.hpp:21:10: warning: enumeration value ‘CUDNN_STATUS_RUNTIME_FP_OVERFLOW’ not handled in switch [-Wswitch]
CXX src/caffe/layers/cudnn_sigmoid_layer.cpp
In file included from ./include/caffe/util/device_alternate.hpp:40:0,
                 from ./include/caffe/common.hpp:19,
                 from ./include/caffe/blob.hpp:8,
                 from ./include/caffe/layers/silence_layer.hpp:6,
                 from src/caffe/layers/silence_layer.cpp:3:
【解决４】
 作者发布的源码中caffe的版本较低，下载最新的caffe，将相关源码覆盖掉ssd中的


 【错误５】
CXX/LD -o .build_release/tools/convert_imageset.bin
//usr/lib/x86_64-linux-gnu/libblas.so.3：对‘gotoblas’未定义的引用
collect2: error: ld returned 1 exit status
Makefile:619: recipe for target '.build_release/tools/upgrade_solver_proto_text.bin' failed
make: *** [.build_release/tools/upgrade_solver_proto_text.bin] Error 1
make: *** 正在等待未完成的任务....
//usr/lib/x86_64-linux-gnu/libblas.so.3：对‘gotoblas’未定义的引用

【解决５】
sudo apt-get install libopenblas-dev
sudo apt-get install libopenblas-base
在makefile.config中加入路径/usr/lib/x86_64-linux-gnu/，具体如下
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/lib/x86_64-linux-gnu/hdf5/serial/include /usr/lib/x86_64-linux-gnu/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/

以下安装不分先后
【安装tensorflow1.13.1】
pip install tensorflow1.13.1*
如果某些依赖的python包无法下载，可去pypi网站上自行下载并安装
下载后的安装方式为pip install **.whl
[tensorflow 1.13 need cuda10.0 cudnn7.4]

【安装mxnet】
由于我安装的是cuda10.0，因此使用如下命令安装
pip install mxnet-cu100


【安装wps】
去官网下载deb格式的安装包，下载后直接双击deb安装包，进入软件管理器自动安装。
安装后报错，系统缺失字体。
解决方法https://www.cnblogs.com/dinphy/p/5888546.html



【安装matlab2017b】
sudo mkdir /media/matlab
sudo mount -o loop ./R2017b_glnxa64_dvd1.iso /media/matlab
cd /
sudo /media/matlab/install
- Install choosing the option "Use a File Installation Key" and supply the following FIK
	09806-07443-53955-64350-21751-41297
先退出第一个盘
sudo mount -o loop ./R2017b_glnxa64_dvd2.iso /media/matlab
激活
进入/usr/local/MATLAB/R2016b/bin
sudo ./matlab激活
进入crack下R2017b/bin/glnxa64拷贝文件
sudo cp license_standalone.lic /usr/local/MATLAB/R2017b/licenses/ 
sudo cp libmwservices.so /usr/local/MATLAB/R2017b/bin/glnxa64/
任意位置终端打开matlab
在目录/usr/local/bin里面创建一个指向Matlab安装目录/usr/local/MATLAB/R2017b/bin的符号链接：（非默认安装需替换安装路径） 
sudo ln -s /usr/local/MATLAB/R2017b/bin/matlab /usr/local/bin/matlab 
权限问题
sudo chmod a+w -R ~/.matlab
