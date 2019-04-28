import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchMatchedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(BatchMatchedMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        """Input and target have shape (batch, output_dim)
        Steps:
        1. Compute a matrix of shape (batch, batch), where each element (i,j) is
           MSE(input_i, target_j)
        2. Take the loss as the row-wise min
        3. Apply reduction
        """
        mse_matrix = input.unsqueeze(1) - target.unsqueeze(0) # compute (B,B,D) difference matrix
        mse_matrix = mse_matrix**2 # Squared differences
        if self.reduction != 'none':
            mse_matrix = mse_matrix.mean(dim=-1) if self.reduction == 'mean' else mse_matrix.sum(dim=-1)
        # Now we have a (B,B) error matrix
        mse_vec_a, _ = mse_matrix.min(dim=1) # row mins (B,)
        mse_vec_b, _ = mse_matrix.min(dim=0) # col mins (B,)
        loss = torch.cat((mse_vec_a, mse_vec_b))
        if self.reduction != 'none':
            loss = loss.mean() if self.reduction == 'mean' else loss.sum()
        return loss

# In this version, we just take min of rows, don't add the col mins
class BatchMatchedMSELoss2(nn.Module):
    def __init__(self, reduction='mean'):
        super(BatchMatchedMSELoss2, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        """Input and target have shape (batch, output_dim)
        Steps:
        1. Compute a matrix of shape (batch, batch), where each element (i,j) is
           MSE(input_i, target_j)
        2. Take the loss as the row-wise min
        3. Apply reduction
        """
        mse_matrix = input.unsqueeze(1) - target.unsqueeze(0) # compute (B,B,D) difference matrix
        mse_matrix = mse_matrix**2 # Squared differences
        mse_matrix = mse_matrix.mean(dim=-1)
        # Now we have a (B,B) error matrix
        loss, _ = mse_matrix.min(dim=1) # row mins (B,)
        #mse_vec_b, _ = mse_matrix.min(dim=0) # col mins (B,)
        #loss = torch.cat((mse_vec_a, mse_vec_b))
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'median':
            loss = loss.median()
        return loss
