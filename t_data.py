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

datasets=[("BeijingAirQuality",7,24,[96,192,336,720],96),
          ("Electricity",321,24,[96,192],96),
          ("ETTh1",7,48,[96,192,336,720],96),
          ("ETTh2",7,24,[96,192,336,720],96),
          ("ETTm1",7,48,[96,192,336,720],96),
          ("ETTm2",7,24,[96,192,336,720],96),
          ("ExchangeRate",8,1,[96,192,336],96),
          ("illness",7,4,[24,36,48,60],36),
        #   ("Traffic",862,24,[96,192,336,720],96),
          ("Weather",21,6*24,[96,192,336,720],96)]

# for dataset,num_features,period_len,output_lens,input_len in datasets: