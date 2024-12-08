# Gradient Loss to just compare the gradients of two spectral curves
class gradientLoss(nn.Module):
    def __init__(self, N, mse_lambda = 2, gradient_lambda = 2e-1):
        super().__init__()
        self.mse_lambda = mse_lambda
        self.gradient_lambda = gradient_lambda
        self.N = N


    def forward(self, pred, gt, epoch=0):

        mse = F.mse_loss(pred, gt) / (self.N*2)

        pred_diff = torch.diff(pred, dim =  1)
        gt_diff = torch.diff(gt, dim = 1)

        pred_diff_flat = pred_diff.view(-1, pred_diff.shape[1])
        gt_diff_flat = gt_diff.view(-1, gt_diff.shape[1])

        # Try other similarity measures! Cosine similarity works best from what I've seen. 
        cosine_sim = torch.cosine_similarity(pred_diff_flat, gt_diff_flat, dim = 1)
        slope_loss = 1-cosine_sim
        slope_loss = slope_loss.mean()/self.N

        if epoch == 0: 
            return self.mse_lambda*mse + self.gradient_lambda*slope_loss
        else:
            # This part is inspired by https://github.com/Liyong8490/HSI-SR-GDRRN. If it doesn't work as expected, try changing the weights or making it epoch-independent. 
            norm = self.mse_lambda + self.gradient_lambda * 0.1 **(epoch//10)
            lamd_slope = self.gradient_lambda * 0.1 ** (epoch // 10)
            total_loss = self.mse_lambda/norm * mse + lamd_slope/norm * slope_loss
            return total_loss
