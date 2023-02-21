import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

# define gradient multiply layer
class GradMultiplyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        # scale only during backward
        return grad_output * ctx.scale, None

class GradMultiplyLayer(torch.nn.Module):
    def __init__(self, scale):
        super(GradMultiplyLayer, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        return GradMultiplyFunction.apply(x, self.scale)

# define model
class PROMET(nn.Module):
    def __init__(self, plm, n_class, input_size=768, lambd=1e-7):
        super(PROMET, self).__init__()
        self.plm = AutoModel.from_pretrained(plm, is_decoder=False, add_pooling_layer=False,
                                                 output_hidden_states=True)
        self.dropout = nn.Dropout(0.1, inplace=False)
        self.gml_layer = GradMultiplyLayer(scale=lambd)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.clf_layer = nn.Linear(input_size, out_features=n_class, bias=True)

    def forward(self, input_ids, mask, mask_ids):
        last_hs = self.plm(input_ids, attention_mask=mask).last_hidden_state
        out = []
        for id, mask_id in enumerate(mask_ids):
            mask_pos = torch.where(mask_id==1)[0]
            filter_hs = last_hs[id, mask_pos, :].T
            filter_hs = filter_hs.unsqueeze(2)
            pooled_hs = self.avg_pool(filter_hs).flatten()
            out.append(pooled_hs)
        out = torch.stack(out)
        out = self.gml_layer(out)
        out = self.dropout(out)
        out = self.clf_layer(out)
        return out