# pylint: disable=wrong-import-position

import os
import sys
import torch

# 方法1.1：设置最大显存限制（单位：字节）
torch.cuda.set_per_process_memory_fraction(0.7)  # 最多使用50%显存
torch.cuda.empty_cache()  # 清空缓存
sys.path.append(os.path.abspath(__file__ + "/../src/"))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from src.basicts import BasicTSLauncher
from src.basicts.configs import BasicTSForecastingConfig



from src.basicts.models.SOFTS import SOFTSConfig, SOFTS

def test_smoke_test(dataset="ETTm1",num_features=7,input_len=96,output_len=32):
    model_config = SOFTSConfig(
        input_len=input_len,
        num_features=num_features,
        output_len=output_len
        )

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=SOFTS,
        model_config=model_config,
        dataset_name=dataset,
        mask_ratio=0.25,
        gpus='0',
        batch_size=16,
        input_len=input_len,
        num_epochs=5,
        output_len=output_len
    ))


from t_data import *
if __name__=='__main__':
    for dataset,num_features,period_len,output_lens,input_len in datasets:
        for output_len in output_lens:
            torch.cuda.empty_cache()
            test_smoke_test(dataset=dataset,num_features=num_features,input_len=input_len,output_len=output_len)
            