import torch


def maskNLLLoss(inp, target, mask, device=None):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    if device:
        loss = loss.to(device)
    return loss, nTotal.item()