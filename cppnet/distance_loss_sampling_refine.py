import torch
import torch.nn.functional as F
from SamplingFeatures import SamplingFeatures


def dice_loss(pred, target, eps=1e-7):
    b = pred.shape[0]
    n_cls = pred.shape[1]
    loss = 0.0
    for ic in range(n_cls):
        ic_target = (target == ic).float().view(b, -1)
        ic_pred = pred[:, ic, :, :].view(b, -1)
        loss += (2*(ic_pred*ic_target).sum(dim=1)+eps) / (ic_pred.pow(2).sum(dim=1)+ic_target.pow(2).sum(dim=1)+eps)
    loss /= n_cls
    loss = 1.0 - loss.mean()
    return loss

class L1Loss_List_withSAP_withSeg(torch.nn.Module):
    def __init__(self, feature_extractor=None, scale=[1,1,1,1]):
        super(L1Loss_List_withSAP_withSeg, self).__init__()
        assert len(scale)==4
        self.scale = scale
        self.feature_extractor = feature_extractor
        if self.scale[3] > 0:
            assert(self.feature_extractor is not None)
    def forward(self, prediction, target_dists, **kwargs):

        prob =  kwargs.get('labels', None)
        pred_dists = prediction[0]
        pred_probs = prediction[1]

        l1loss = 0.0
        bceloss = 0.0
        
        for i_dist in pred_dists:
            l1loss_map = F.l1_loss(i_dist, target_dists, reduction='none')
            l1loss += torch.mean(prob*l1loss_map)
        for i_prob in pred_probs:
            bceloss += F.binary_cross_entropy(i_prob, prob)

        loss = self.scale[0]*l1loss + self.scale[1]*bceloss

        if self.scale[2] > 0:
            segloss = 0.0
            pred_segs = prediction[2]
            seg = (prob>0).float()
            for i_seg in pred_segs:
                segloss += F.binary_cross_entropy(i_seg, seg)
            loss += self.scale[2]*segloss

        metric = loss.data.clone().cpu()

        if self.scale[3] > 0:
            self.feature_extractor.zero_grad()
            sap_loss = 0.0
            f_target = self.feature_extractor(torch.cat((prob, target_dists), dim=1))
            for i_dist in pred_dists:
                f_pred = self.feature_extractor(torch.cat((pred_probs[-1]*pred_segs[-1], i_dist*pred_segs[-1]), dim=1))
                sap_loss += F.l1_loss(f_pred, f_target)
            loss += self.scale[3]*sap_loss

        # print('loss: {:.5f}, metric: {:.5f}, l1: {:.5f}, bce: {:.5f}, seg: {:.5f}, sap: {:.5f}'\
        #     .format(loss.item(), metric.item(), l1loss.item(), bceloss.item(), segloss.item(), sap_loss.item()))

        return loss, metric