import torch
import torch.nn.functional as F

class InstanceLoss(torch.nn.Module):
    def __init__(self,scale=[1, 1]):
        super(InstanceLoss, self).__init__()
        assert len(scale)==2
        self.scale = scale
    def forward(self, prediction, gt_segbnd, **kwargs):
    
        segbnd = prediction[0]
        bbox = prediction[1]
        gt_bbox = kwargs['bbox']
        gt_seg = gt_segbnd[:, 0]
        if self.scale[1]>0:
            bboxloss = F.l1_loss(bbox, gt_bbox, size_average=False, reduce=False)*gt_seg.unsqueeze(dim=1)
            bboxloss = torch.mean(bboxloss)
        else:
            bboxloss = 0.0
        if self.scale[0]>0:
            segbndloss = F.binary_cross_entropy(segbnd, gt_segbnd, weight=None, size_average=True, reduce=True)
        else:
            segbndloss = 0.0

        return self.scale[0]*segbndloss + self.scale[1]*bboxloss
