# pylint: disable=wrong-import-position
from t_data import *
import os
import sys
import torch

# 方法1.1：设置最大显存限制（单位：字节）
torch.cuda.set_per_process_memory_fraction(0.8)  
torch.cuda.empty_cache()  # 清空缓存
sys.path.append(os.path.abspath(__file__ + "/../src/"))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.basicts import BasicTSLauncher
from src.basicts.configs import BasicTSForecastingConfig



from src.basicts.models.SparseTSF import SparseTSFConfig, SparseTSF

def test_smoke_test(dataset="ETTm1",num_features=7,input_len=96,output_len=32,period_len=96):
    model_config = SparseTSFConfig(
        input_len=input_len,
        period_len=period_len,#日周期
        output_len=output_len
        )

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=SparseTSF,
        model_config=model_config,
        dataset_name=dataset,
        mask_ratio=0.25,
        gpus='0',
        batch_size=16,
        input_len=input_len,
        num_epochs=5,
        output_len=output_len
    ))




if __name__=='__main__':
    # test_smoke_test(dataset='Weather',num_features=21,input_len=6*24,output_len=336,period_len=6*24)
    for dataset,num_features,period_len,output_lens,input_len in datasets:
        if dataset =='Weather':
            continue
        for output_len in output_lens:
            torch.cuda.empty_cache()
            test_smoke_test(dataset=dataset,num_features=num_features,input_len=input_len,output_len=output_len,period_len=period_len)
 