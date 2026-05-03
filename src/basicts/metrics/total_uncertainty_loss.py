# custom_loss.py
import torch
def total_uncertainty_loss(prediction, targets, targets_mask=None, log_variance=None, lambda_unc=0.1):
    """
    总损失函数 - 公式 (4.43)
    L_total = L_MAE + λ * L_unc
    
    Args:
        prediction: 预测均值 μ
        targets: 真实值 Y
        targets_mask: 掩码
        log_variance: 对数方差 log σ²
        lambda_unc: 不确定性损失权重 λ
    """
    # if isinstance(prediction, dict):
    #     log_variance = prediction['log_variance']
    #     prediction = prediction['prediction']
    # 应用掩码
    if targets_mask is not None:
        valid_mask = targets_mask.bool()
        prediction = prediction[valid_mask]
        targets = targets[valid_mask]
        log_variance = log_variance[valid_mask]
    
    # 公式 (4.42): L_MAE
    mae_loss = torch.mean(torch.abs(prediction - targets))
    
    # 公式 (4.38): L_unc
    variance = torch.exp(log_variance)
    squared_error = (targets - prediction) ** 2
    unc_loss = 0.5 * torch.mean(squared_error / variance + log_variance)
    
    # 公式 (4.43): L_total
    total_loss = mae_loss + lambda_unc * unc_loss
    
    return total_loss