from t_data import * #datasets

from src.basicts.models.myModel import myModel,MyModelConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

def test_smoke_test(dataset="ETTm1",num_features=7,input_len=96,output_len=32,period_len=96):
    model_config = MyModelConfig(
        input_len=input_len,
        output_len=output_len,
        num_features=num_features,
        )

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=myModel,
        model_config=model_config,
        dataset_name=dataset,
        mask_ratio=0.25,
        gpus='0',
        batch_size=16,
        input_len=input_len,
        num_epochs=50,
        output_len=output_len,

    ))

if __name__=='__main__':
    # test_smoke_test(dataset="illness",num_features=7,input_len=36,output_len=24,period_len=4)
    for dataset,num_features,period_len,output_lens,input_len in datasets:
        if dataset=='Electricity':
            continue
        for output_len in output_lens:
            torch.cuda.empty_cache()
            test_smoke_test(dataset=dataset,num_features=num_features,input_len=input_len,output_len=output_len,period_len=period_len)
            