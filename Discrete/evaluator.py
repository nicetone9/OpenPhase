import torch
import torch.nn.functional as F
import numpy as np

class Evaluator(object):
    def __init__(self, blosum_path="/home/peiranj/mapdiff/eval_blosums.pth", blosum_eval=True, reorder_blosum=False,
                 reorder_types=None):
        self.blosum_names = ['BLOSUM45', 'BLOSUM62', 'BLOSUM80', 'BLOSUM90']
        self.blosum_aa_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                                'Y', 'V']
        self.blosum_eval = blosum_eval
        self.reorder_types = reorder_types
        if self.blosum_eval:
            self.blosum_mats = torch.load(blosum_path)
            if self.reorder_types is not None and reorder_blosum:
                blosum_aa_dict = {aa: i for i, aa in enumerate(self.blosum_aa_order)}
                new_order = [blosum_aa_dict[aa] for aa in reorder_types]
                for blosum_name in self.blosum_names:
                    blosum_mat = self.blosum_mats[blosum_name]
                    re_blosum_mat = blosum_mat[np.ix_(new_order, new_order)]
                    self.blosum_mats[blosum_name] = re_blosum_mat

    def cal_blosum_nssr(self, pred_seq, target_seq, blosum_name):
        scores = self.blosum_mats[blosum_name][pred_seq, target_seq] > 0
        nssr = (scores.sum() / pred_seq.shape[0]).item()
        return nssr

    def cal_all_blosum_nssr(self, pred_seq, target_seq):
        nssr_dict = {}
        for blosum_name in self.blosum_mats.keys():
            nssr = self.cal_blosum_nssr(pred_seq, target_seq, blosum_name)
            nssr_dict[blosum_name] = nssr
        return [nssr_dict[blosum_name] for blosum_name in self.blosum_names]

    @staticmethod
    def cal_recovery(pred_seq, target_seq):
        recovery = (pred_seq == target_seq).sum() / pred_seq.shape[0]
        recovery = recovery.item()
        return recovery

    @staticmethod
    def cal_perplexity(pred_logits, target_label):
        ll_fullseq = F.cross_entropy(pred_logits, target_label, reduction='mean').item()
        perplexity = np.exp(ll_fullseq)
        return perplexity