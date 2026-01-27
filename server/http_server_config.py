# coding: utf-8
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class BasicConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='bts_server_')

class ServerConfig(BasicConfig):
    host: str = '0.0.0.0'
    port: int = 8502

class ModelConfig(BasicConfig):
    cfg_path: str = 'src/basicts/models/Crossformer/config/crossformer_config.py'
    ckpt_path: str = 'checkpoints/Crossformer/BeijingAirQuality_5_96_96/a6d48024fc3d05a191472786c3f61e90/Crossformer_best_val_MAE.pt'
    device_type: str = 'gpu'
    gpus: Optional[str] = '0'
    context_length: int = 72
    prediction_length: int = 24
    