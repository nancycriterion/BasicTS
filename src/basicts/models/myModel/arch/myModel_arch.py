import torch
from ..config.myModel_config import MyModelConfig
from torch import nn
from .myModel_layers import EDM,GCA2res_add,GASF,CausalPredictor,GTU
from basicts.modules import MLPLayer

class myModel(nn.Module):
    def __init__(self,config:MyModelConfig):
        super(myModel,self).__init__()
        # self.ukf=MyUKF(config)
        self.out_center_EDM=EDM(num_layers2imf=config.num_layers2imf,
                                method=config.centor_method)
        self.gasf=GASF()
        self.GCA=[]
        
        self.GCA.append(GCA2res_add(cnn_out_channels=config.cnn_out_channels,
                                    hidden_layers=config.hidden_layers,
                                    seq_len=config.input_len,
                                    num_features=config.num_features,
                                    cnn_in_channels=config.num_layers2imf))
        self.GCA=nn.ModuleList(self.GCA)
        self.CaPre=CausalPredictor(num_features=config.num_features,
                                   n_heads=2,
                                   out_len=config.output_len,
                                   seq_len=config.input_len)
        self.trend_DMS=nn.Sequential(*[MLPLayer(config.input_len,config.input_len) for _ in range(config.nums_DMS)],
                                        nn.Linear(config.input_len,config.output_len))
        self.res_cnn=nn.Sequential(
            nn.Conv1d(config.input_len,config.input_len*2,kernel_size=17,padding=8),
            nn.ReLU(),
            nn.Conv1d(config.input_len*2,config.input_len*4,kernel_size=17,padding=8),
            nn.ReLU(),
            nn.Conv1d(config.input_len*4,config.input_len*8,kernel_size=17,padding=8),
        )
        self.res_gca_linear=nn.Linear((1+1)*config.input_len,config.input_len*8)
        self.GTUs=[]
        hidden_size = [config.input_len*8]
        hidden_size.extend(config.hidden_sizes)
        for i in range(1,len(hidden_size)):
            self.GTUs.append(GTU(seq_len=hidden_size[i-1],hidden_len=hidden_size[i],dropout=config.dropout))
        self.GTUs=nn.ModuleList(self.GTUs)
        self.GTU_linear=nn.Linear(config.hidden_sizes[-1],config.output_len)
        self.linear=nn.Linear(3,1)
    
    def forward(self,inputs:torch.Tensor)->torch.Tensor:
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, input_len, num_features]

        Returns:
            torch.Tensor: Output tensor with shape [batch_size, output_len, num_features]
        """
        
        batch_size, input_len, num_features = inputs.shape
        imfs,residue,trend_pre=self.out_center_EDM(inputs)#[batch_size,max_imfs,seq_len,num_features],[batch_size,seq_len,num_features]
        device = next(self.parameters()).device
        residue=residue.to(device)
        trend_pre=trend_pre.to(device)
        imfs=imfs.to(device)
        gasf_images=self.gasf(imfs)#[batch, n_imfs, seq_len, seq_len, features]
        gca_gasf,res_gca=self.GCA[0](gasf_images)#[batch_size,2,seq_len,num_features]
        cp_gca=self.CaPre(gca_gasf).squeeze(1)#[batch_size,output_len,num_features]
        trend_dms=self.trend_DMS(trend_pre.permute(0,2,1)).permute(0,2,1)#[batch_size,output_len,num_features]
        res_cnn=self.res_cnn(residue)#[batch_size,seq_len*8,num_features]
        res_gca=res_gca.reshape(batch_size,-1,num_features).permute(0,2,1)#[batch_size,num_features,(1+1)*seq_len]
        res_gca=self.res_gca_linear(res_gca).permute(0,2,1)#[batch_size,seq_len*8,num_features]
        res=res_cnn+res_gca
        for gtu in self.GTUs:
            res=gtu(res)#[batch_size,hidden_size[-1],num_features]
        res=self.GTU_linear(res.permute(0,2,1)).permute(0,2,1)#[batch_size,output_len,num_features]
        # print(cp_gca.shape,trend_dms.shape,res.shape)
        output=torch.concat([cp_gca.unsqueeze(3),trend_dms.unsqueeze(3),res.unsqueeze(3)],dim=3)#[batch_size,output_len,num_features,3]
        output=self.linear(output).squeeze(3)#[batch_size,output_len,num_features]
        return output