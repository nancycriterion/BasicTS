import numpy as np
import torch.nn as nn
import torch
from PyEMD import EMD
from basicts.modules.transformer import MultiHeadAttention

class MyUKF(nn.Module):
    def __init__(self):
        super(MyUKF, self).__init__()
        """
        used for howl model
        """
        
class EDM(nn.Module):
    def __init__(self,num_layers2imf,method='linear'):
        """
        __init__ 的 Docstring
        
        :param self: 说明
        :param num_layers2imf: 说明
        :param method: 说明:'mean' ,'linear','moving_avg'
        """
        super(EDM, self).__init__()
        self.max_imfs = num_layers2imf
        self.method=method
    
    def forward(self, x):
        """
        forward 的 Docstring
        
        :param self: 说明
        :param x: 说明
        
        :return:
        :imfs:[batch_size,max_imfs,seq_len,num_features]
        :residue:[batch_size,seq_len,num_features]
        :trend_pre:[batch_size,seq_len,num_features]
        """
        batch_size,seq_len,num_features = x.shape
        #去中心化，分解，输出imfs和trend。
        detrended_tensor,trend_pre_tensor=self._batch_detrend(x)
        imfs_tensor,residue_tensor=self._edm_div(detrended_tensor)
        return imfs_tensor,residue_tensor,trend_pre_tensor
        
        
    def _batch_detrend(self, x):
        """
        批量去中心化
        
        :param self: 说明
        :param x: 说明
        """
        batch_size,seq_len,num_features = x.shape
        if self.method =='mean':
            means=x.mean(dim=1,keepdim=True)#[batch_size,1,num_features]
            detrended=x-means
            trend_pre=means.repeat(1,seq_len,1)
        elif self.method == 'linear':
            detrended=torch.zeros_like(x)
            trend_pre=torch.zeros_like(x)
            t=torch.arange(seq_len,dtype=torch.float32)
            for b in range(batch_size):
                for f in range(num_features):
                    series=x[b,:,f]
                    A=torch.stack([t,torch.ones_like(t)],dim=1)
                    coeff=torch.linalg.lstsq(A,series.unsqueeze(1)).solution
                    linear_trend=coeff[0]*t+coeff[1]
                    trend_pre[b,:,f]=linear_trend
                    detrended[b,:,f]=series-linear_trend
        elif self.method =='moving_avg':
            window_size=min(31,seq_len//10)#自适应窗口
            if window_size%2==0:
                window_size+=1
            kernel=torch.ones(1,1,window_size)/window_size
            trends_pre=[]
            for f in range(num_features):
                feature_data=x[:,:,f].unsqueeze(1)# [batch_size,1,seq_len]
                #使用反射填充处理边界
                padding=window_size//2
                padded=torch.nn.functional.pad(feature_data,
                                               (padding,padding),
                                               mode='replicate')
                ma=torch.nn.functional.conv1d(
                    padded,
                    kernel,
                    padding=0
                ).squeeze(1)
                trend_pre.append(ma)
            trend_pre=torch.stack(trends_pre,dim=2)
            detrended=x-trend_pre
        return detrended,trend_pre
    def _edm_div(self, x):
        x = x.cpu().numpy()
        batch_size, seq_len, num_features = x.shape
        emd=EMD()
        all_imfs=[]
        all_residues=[]
        
        for b in range(batch_size):
            batch_imfs = []  # 这个批次的 IMF
            batch_residues = []  # 这个批次的残差
            for f in range(num_features):
                # 提取单个序列
                series = x[b, :, f]
                
                # 执行 EMD 分解
                emd(series,max_imf=self.max_imfs)
                imfs,residue=emd.get_imfs_and_residue()# [n_imfs, seq_len]
                n_imfs = imfs.shape[0]
        
        # 处理 IMF 数量不够 max_imf 的情况
                if n_imfs < self.max_imfs:
                    # 补零
                    padded = np.zeros((self.max_imfs, seq_len))
                    padded[:n_imfs, :] = imfs
                    imfs = padded
                    
                # print(f,':',imfs.shape,residue.shape)
                batch_imfs.append(imfs)
                batch_residues.append(residue)
                
            batch_imfs = np.array(batch_imfs)  # [features, n_imfs, seq_len]
            batch_imfs = np.transpose(batch_imfs, (1, 2, 0))  # [n_imfs, seq_len, features]
            batch_residues = np.array(batch_residues).T  # [seq_len, features]
            
            all_imfs.append(batch_imfs)
            all_residues.append(batch_residues)
        imfs_tensor = torch.tensor(np.array(all_imfs), dtype=torch.float32)
        residue_tensor = torch.tensor(np.array(all_residues), dtype=torch.float32)
        return imfs_tensor, residue_tensor

class GCA2res_add(nn.Module):
    def __init__(self,cnn_out_channels,hidden_layers,seq_len,num_features,cnn_in_channels=1):
        super(GCA2res_add, self).__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(cnn_in_channels, 8, kernel_size=3, padding=1),  # padding=1 保持 T
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(0.1),

            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),

            nn.Conv2d(16, cnn_out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        self.linear_layers=nn.Linear(cnn_out_channels,1)
        reduction_factor = 4
        reduced_dim = seq_len // reduction_factor  # 96/4=24
        self.linear_layers2 = nn.Sequential(
            nn.Linear(seq_len*seq_len, reduced_dim),  # 9216→24
            nn.ReLU(),
            nn.Linear(reduced_dim, seq_len),  # 24→96
            nn.LayerNorm(seq_len),
        )

        self.linear_layers3 = nn.Sequential(
            nn.Linear(seq_len, reduced_dim),  # 96→24
            nn.ReLU(), 
            nn.Linear(reduced_dim, seq_len*seq_len),  # 24→9216
            nn.LayerNorm(seq_len*seq_len),
        )

        self.q_train=nn.Parameter(torch.randn(2,seq_len,seq_len,1))#可训练的向量
        self.se_attn=nn.ModuleList(MultiHeadAttention(hidden_size=cnn_out_channels,n_heads=4) for _ in range(num_features))
        self.attn_FFNs=[]
        for _ in range(hidden_layers):
            self.attn_FFNs.append(across_self_attention(num_features,seq_len,seq_len))
        self.attn_FFNs=nn.ModuleList(self.attn_FFNs)
    def forward(self, x):
        #x.shape=[batch_size,in_channel=cnn_in_channels,seq_len,seq_len,num_features]
        _,_,seq_len,_,num_features=x.shape
        device = next(self.parameters()).device  # 获取模型所在设备
        x = x.to(device)  # 把输入搬到模型同一设备
        B, C, T, T, F = x.shape
        x_reshape = x.permute(0, 4, 1, 2, 3).reshape(B*F, C, T, T)
        out = self.conv_layers(x_reshape)  # [B*F, out_channel, T, T]
        x = out.view(B, F, -1, T, T).permute(0, 2, 3, 4, 1)  # [B, out_channel, T, T, F]
        #x.shape=[batch_size,out_channel=cnn_out_channels,seq_len,seq_len,num_features]
        x_cross=self.linear_layers(x.permute(0,4,2,3,1)).permute(0,4,2,3,1)
        #x_cross.shape=[batch_size,1,seq_len,seq_len,num_features]
        x=x.permute(0,2,3,1,4)#x.shape=[batch_size,seq_len,seq_len,out_channel=cnn_out_channels,num_features]
        x=x.view(B,T*T,-1,F).permute(0,2,3,1)
        x=self.linear_layers2(x).permute(0,3,1,2)#减少内存占用
        x_cross=self.linear_layers2(x_cross.view(B,1,T*T,F).permute(0,1,3,2)).permute(0,1,3,2)
        temp=[]
        for feature in range(num_features):
            out,w,_=self.se_attn[feature](x[:,:,:,feature])
            # attn_mean = w.mean(dim=1)  # [B, 1, L]
            out=self.linear_layers(out)
            temp.append(out)
        x=torch.stack(temp,dim=3)#x.shape=[batch_size,seq_len,1,num_features]
        x=x.permute(0,2,1,3)#x.shape=[batch_size,1,seq_len,num_features]
        x=torch.cat([x_cross,x],dim=1)
        #x.shape=[batch_size,2,seq_len,num_features]
        x=self.linear_layers3(x.permute(0,1,3,2)).permute(0,1,3,2).reshape(B,2,T,T,F)
        #x.shape=[batch_size,2,seq_len,seq_len,num_features]
        for attn_FFN in self.attn_FFNs:
            x_attn=attn_FFN(x,self.q_train)
        #x.shape=[batch_size,2,seq_len,num_features]
        res=x-x_attn
        res=self.linear_layers2(res.view(B,2,T*T,F).permute(0,1,3,2)).permute(0,1,3,2)
        x_attn=self.linear_layers2(x_attn.view(B,2,T*T,F).permute(0,1,3,2)).permute(0,1,3,2)
        return x_attn,res
class across_self_attention(nn.Module):
    def __init__(self,num_features,seq_len,hidden_size,n_heads=2):
        super(across_self_attention, self).__init__()
        self.num_features=num_features
        self.seq_len=seq_len
        self.self_attn=MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads
        )
        self.across_attn=FeatureCrossAttention(d_k=16)
        self.linear=nn.Linear(num_features+1,num_features)
        self.linear_layers2=nn.Linear(seq_len*seq_len,seq_len)
        self.linear_layers3=nn.Linear(seq_len,seq_len*seq_len)
        
    def forward(self,x,q):
        x_cross=x[:,0]
        x_self=x[:,1]
        B, L,L, F = x_self.shape
        q = q.unsqueeze(0)      # (1, 2, L,L, 1)
        q = q.expand(B, -1, -1, -1,-1)         # (B, 2,L, L, 1)
        output_self=[]
        for f in range(F):
            x_self_f=x_self[:,:,:,f]#[batch_size，L,L]
            # x_self_f=x_self_f.permute(0,2,1)
            out,_,_=self.self_attn(x_self_f)
            out=out.permute(0,2,1)
            output_self.append(out)
        output_self=torch.stack(output_self,dim=3)
        x_self=output_self
        #output_self.shape=[batch_size,cnn_out_channels,seq_len,num_features]
        x_cross=self.linear_layers2(x_cross.view(B,L*L,F).permute(0,2,1)).permute(0,2,1)
        #x_cross.shape=[batch_size,seq_len,num_features]
        x_cross=self.across_attn(x_cross)
        x_cross=self.linear_layers3(x_cross.permute(0,2,1)).permute(0,2,1).view(B,L,L,F)
        #x_cross.shape=[batch_size,seq_len,seq_len,num_features]
        x=torch.concat([x_cross.unsqueeze(1),x_self.unsqueeze(1)],dim=1)
        #x.shape=[batch_size,2,seq_len,seq_len,num_features]
        
        x=torch.concat([x,q],dim=4)
        #x.shape=[batch_size,2,seq_len,seq_len,num_features+1]
        x=self.linear(x)
        #x.shape=[batch_size,2,seq_len,seq_len,num_features]
        return x
        

class FeatureCrossAttention(nn.Module):
    def __init__(self, d_k=16):
        super().__init__()
        self.q_proj = nn.Linear(1, d_k)
        self.k_proj = nn.Linear(1, d_k)
        self.v_proj = nn.Linear(1, 1)
        self.scale = d_k ** 0.5

    def forward(self, x):
        """
        x: [B, L, F]
        return: [B, L, F]
        """
        B, L, F = x.shape

        # Treat (B,L) as batch
        x_ = x.reshape(B * L, F).unsqueeze(2)  # [BL, F, 1]

        Q = self.q_proj(x_)        # [BL, F, d_k]
        K = self.k_proj(x_)        # [BL, F, d_k]
        V = self.v_proj(x_)        # [BL, F, 1]

        # Feature-wise attention
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # softmax over features

        out = torch.matmul(attn_weights, V)  # [BL, F, 1]

        return out.view(B, L, F)
        
    
    
    
class GASF(nn.Module):
    def __init__(self,method='summation'):
        super(GASF, self).__init__()
        self.method=method
        
    def forward(self, x):
        """处理 4D 张量: [batch, n_imfs, seq_len, features]"""
        batch_size, n_imfs, seq_len, n_features = x.shape
        
        # 结果形状: [batch, n_imfs, seq_len, seq_len, features]
        gasf_images = torch.zeros(
            batch_size, n_imfs, seq_len, seq_len, n_features,
            device=x.device, dtype=x.dtype
        )
        
        for b in range(batch_size):
            for imf_idx in range(n_imfs):
                for f in range(n_features):
                    imf_series = x[b, imf_idx, :, f]
                    gasf = self._imf_to_gasf(imf_series)
                    gasf_images[b, imf_idx, :, :, f] = gasf
        
        return gasf_images
    
    def _imf_to_gasf(self, imf_series):
        
        #归一化
        # 方法1：基于标准差的比例缩放
        std = torch.std(imf_series)
        if std > 0:
            # 缩放到标准差为0.5（这样大部分值在[-1,1]内）
            scaled = imf_series / (std * 2)
        else:
            scaled = imf_series
        scaled = torch.clamp(scaled, -1 + 1e-8, 1 - 1e-8)
        # 角度变换
        phi = torch.arccos(scaled)
        
        # 构建 GASF/GADF
        if self.method == 'summation':
            # GASF: cos(φ_i + φ_j)
            phi_i = phi.unsqueeze(1)  # [seq_len, 1]
            phi_j = phi.unsqueeze(0)  # [1, seq_len]
            gasf = torch.cos(phi_i + phi_j)
        else:
            # GADF: sin(φ_i - φ_j)
            phi_i = phi.unsqueeze(1)
            phi_j = phi.unsqueeze(0)
            gasf = torch.sin(phi_i - phi_j)
        
        return gasf
    
    
# class CausalPredictor(nn.Module):
#     def __init__(self, num_features, n_heads, hidden_mult=2,out_len=96):
#         super().__init__()
#         self.out_len=out_len
#         self.C = 2
#         self.F = num_features
#         self.hidden = self.C * self.F * hidden_mult

#         self.in_proj = nn.Linear(self.C * self.F, self.hidden)

#         self.self_attn = MultiHeadAttention(
#             hidden_size=self.hidden,
#             n_heads=n_heads
#         )

#         self.out_proj = nn.Linear(self.hidden, self.F)
#     def causal_mask(self, seq_len, device):
#         return torch.triu(
#             torch.full((seq_len, seq_len), float("-inf"), device=device),
#             diagonal=1
#         ).unsqueeze(0).unsqueeze(0)
#     def forward(self, x):
#         """
#         x: [B, 2, L, F]
#         return: [B, out_len, F]
#         """
#         B, C, L, F = x.shape
#         device = x.device

#         # ---- flatten channel ----
#         x = x.permute(0, 2, 1, 3).contiguous()   # [B, L, 2, F]
#         x = x.view(B, L, 2 * F)                  # [B, L, 2F]
#         x = self.in_proj(x)                      # [B, L, hidden]

#         # ===== 1. prefill =====
#         mask = self.causal_mask(L, device)
#         from basicts.modules.transformer.attentions import PastKeyValue
#         past_kv = PastKeyValue()
#         h, _, past_kv = self.self_attn(
#             hidden_states=x,
#             attention_mask=mask,
#             use_cache=True,
#             past_key_value=past_kv,
#             layer_idx=0
#         )

#         outputs = []

#         # ===== 2. decode =====
#         last_h = h[:, -1:]   # [B, 1, hidden]

#         for _ in range(self.out_len):
#             y_t = self.out_proj(last_h.squeeze(1))   # [B, F]
#             outputs.append(y_t)

#             # fake 2-channel
#             y_2c = y_t.unsqueeze(1).repeat(1, 2, 1)  # [B, 2, F]
#             y_2c = y_2c.view(B, 1, 2 * F)            # [B, 1, 2F]
#             y_embed = self.in_proj(y_2c)             # [B, 1, hidden]

#             last_h, _, past_kv = self.self_attn(
#                 hidden_states=y_embed,
#                 past_key_value=past_kv,
#                 use_cache=True,
#                 layer_idx=0
#             )

#         return torch.stack(outputs, dim=1)            # [B, out_len, F]

    
class CausalPredictor(nn.Module):
    def __init__(self, num_features, n_heads, out_len, seq_len=96):
        super().__init__()

        self.C = 2
        self.F = num_features
        self.out_len = out_len

        hidden_size = self.C * self.F

        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            n_heads=n_heads
        )

        # 把 channel 压缩到 1
        self.channel_proj = nn.Linear(hidden_size, self.F)
        self.linear = nn.Linear(seq_len, out_len)
        
    def causal_mask(self,seq_len, device):
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
    def forward(self, x):
        """
        x: [B, 2, L, F]
        return: [B, 1, out_len, F]
        """
        B, C, L, F = x.shape

        assert C == self.C and F == self.F

        # ---- reshape → attention ----
        x = x.permute(0, 2,1,3).contiguous()   # [B, L, C, F]
        x = x.view(B, L, C * F)                  # [B, L, hidden]

        # ---- causal mask ----
        mask = self.causal_mask(L, x.device)
      
        
        # ---- causal self-attention ----
        x, _, _ = self.self_attn(
            hidden_states=x,
            attention_mask=mask
        )  # [B, L, hidden]

        # ---- channel 压缩 ----
        x = self.channel_proj(x)                 # [B, L, F]

        # ---- 取预测窗口 ----
        x=self.linear(x.permute(0,2,1)).permute(0,2,1)           # [B, out_len, F]

        # ---- reshape 输出 ----
        x = x.unsqueeze(1)                       # [B, 1, out_len, F]
        return x

class GTU(nn.Module):
    def __init__(self,seq_len,hidden_len,dropout=0.1):
        super(GTU, self).__init__()
        self.seq_len=seq_len
        
        self.conv_filter=nn.Conv1d(in_channels=seq_len,out_channels=hidden_len,kernel_size=7,padding=3)
        self.conv_gate=nn.Conv1d(in_channels=seq_len,out_channels=hidden_len,kernel_size=7,padding=3)
        
        
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        f=self.conv_filter(x)
        g=self.conv_gate(x)
        h=f*g   
        h=self.dropout(h)

        return h