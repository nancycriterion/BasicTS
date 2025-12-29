from t_data import * #datasets

from src.basicts.models.NLinear import NLinear, NLinearConfig

def test_smoke_test(dataset="ETTm1",num_features=7,input_len=96,output_len=32,period_len=96):
    model_config = NLinearConfig(
        input_len=input_len,
        output_len=output_len
        )

    BasicTSLauncher.launch_training(BasicTSForecastingConfig(
        model=NLinear,
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
    for dataset,num_features,period_len,output_lens,input_len in datasets:
        for output_len in output_lens:
            torch.cuda.empty_cache()
            test_smoke_test(dataset=dataset,num_features=num_features,input_len=input_len,output_len=output_len,period_len=period_len)
            