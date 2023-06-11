import torch
import torch.nn.functional as F

class InstanceLoss(torch.nn.Module):
    def __init__(self,scale=[1, 1], n_cls=0):
        super(InstanceLoss, self).__init__()
        
        self.scale = scale
        self.n_cls = n_cls
        if self.n_cls <= 1:
            assert len(scale) == 2
        else:
            assert len(scale) == 3

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
        loss = self.scale[0]*segbndloss + self.scale[1]*bboxloss

        if self.n_cls > 1:
            gt_cls = gt_segbnd[:, 2:]
            pred_cls_log = prediction[2].log()
            if self.scale[2]>0:
                clsloss = F.kl_div(pred_cls_log, gt_cls, size_average=True, reduce=True)
            else:
                clsloss = 0.0
            loss += self.scale[2]*clsloss

        print('loss: {:.5f}, segbndloss: {:.5f}, bboxloss: {:.5f}, '\
            .format(loss, segbndloss, bboxloss, ))

        return loss

