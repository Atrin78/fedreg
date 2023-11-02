import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).to(logits.device)
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits  

class NTD_Loss(nn.Module):
    """Not-true Distillation Loss"""

    def __init__(self, num_classes=10, tau=3, beta=0):
        super(NTD_Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        self.num_classes = num_classes
        self.tau = tau
        self.beta = beta

    def forward(self, logits, targets, dg_logits):
        if len(logits.shape) != len(targets.shape):
            targets = nn.functional.one_hot(targets.long(), self.num_classes).float()
        ce_loss = -targets*torch.log(logits+1e-12)
        ce_loss = ce_loss.sum(1)
        ntd_loss = self._ntd_loss(logits, dg_logits, targets)
        logger.info(f"ntd_loss: {ntd_loss}")

        loss = ce_loss.mean() + (self.beta * ntd_loss).mean() 


        return loss

    def _ntd_loss(self, logits, dg_logits, targets):
        """Not-tue Distillation Loss"""

        # Get smoothed local model prediction
        logits = refine_as_not_true(logits, targets, self.num_classes)
        pred_probs = F.log_softmax(logits / self.tau, dim=1)

        # Get smoothed global model prediction
        with torch.no_grad():
            dg_logits = refine_as_not_true(dg_logits, targets, self.num_classes)
            dg_probs = torch.softmax(dg_logits / self.tau, dim=1)

        loss = (self.tau ** 2) * self.KLDiv(pred_probs, dg_probs)

        return loss