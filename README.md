## DR3DF-Net

### Dependences
1.Pytorch 1.8.0  
2.Python 3.7.1  
3.CUDA 11.7  
4.Ubuntu 18.04    

### Datasets Preparation
> ./dataset/dataset_name/train
>> clean  
>> hazy

> ./dataset/dataset_name/test 
>> clean  
>> hazy

> ./output_result

### Pretrained Weights and Dataset  
Download our model weights on Baidu cloud disk:  
[https://pan.baidu.com/s/1EmRjYgxeKD8NNLdFMnkDhA](https://pan.baidu.com/s/1uBqLC8pTnRdUGs3J7roVpw) password:`wxan`

Download our test datasets on Baidu cloud disk:  
https://pan.baidu.com/s/1ZvaeTOzJ1fZI41TItf0V5A password:`xpwr`

### 1. Train Deahzing  
 `python mytrain_dehazing.py --type 1 -train_batch_size 4 --gpus 0 `
### 2. Train SCFR 
 `python mytrain_select.py --type 1 -train_batch_size 4 --gpus 0 `

### Test
`python mytest_select.py --type 1 --gpus 0 `


