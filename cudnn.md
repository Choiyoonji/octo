# cuda-12.2 -> 사용할 cuda 버전에 맞게 설정
# so.8.9.6 -> 사용할 cudnn 버전에 맞게 설정

# cp : 복사 명령어
sudo cp cudnn-linux-x86_64-8.9.6.50_cuda12-archive/include/cudnn* /usr/local/cuda-12.2/include
sudo cp cudnn-linux-x86_64-8.9.6.50_cuda12-archive/lib/libcudnn* /usr/local/cuda-12.2/lib64

# chmod : 권한 부여 명령어
sudo chmod a+r /usr/local/cuda-12.2/include/cudnn.h /usr/local/cuda-12.2/lib64/libcudnn*


sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8.9.6 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_adv_train.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8.9.6  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_ops_infer.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8.9.6  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_cnn_train.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8.9.6  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_adv_infer.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8.9.6  /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_ops_train.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8.9.6 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn_cnn_infer.so.8
sudo ln -sf /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn.so.8.9.6 /usr/local/cuda-12.2/targets/x86_64-linux/lib/libcudnn.so.8

sudo ldconfig

ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn