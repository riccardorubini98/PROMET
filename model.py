import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from data import data_prompt_loader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score

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
class promet_clf(nn.Module):
    def __init__(self, plm, n_class, input_size=768, lambd=1e-7):
        super(promet_clf, self).__init__()
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
    
# define main class
class Promet(object):
    
    def __init__(self, plm_name='roberta-base', max_lenght=128, 
                 mask_token='<mask>', n_mask=1, model_name='model', save_dir='/results'):
        # create template
        self.template = '{mention} is a ' + (mask_token+ ' ')*n_mask + '.'
        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(plm_name, truncation=True, truncation_side='left', max_lenght=max_lenght)
        # define other variable
        self.max_lenght = max_lenght
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.save_dir = save_dir
        
    def prompt_tokenizer(self, example):
        """ add prompt template to example and then tokenize
        
        args:
            - example: example from output list of import examples
            - template: prompt template with {mention} and mask tokens
            - tokenizer: hf tokenizer
            - max_lenght
            
        return
            - features: token_ids and attention mask
            - mask_id: a torch.tensor of lenght=max_lenght with 1 for mask token position
            - ex_id
        """
        # build prompt
        prompt = self.template.format(mention=example['mention'])
        # generate example
        text = example['text'].strip('.') + '. ' + prompt
        # tokenize
        features = self.tokenizer.encode_plus(text, return_tensors="pt", truncation=True, 
                                        max_length=self.max_lenght, padding='max_length')
        tokenized_text = self.tokenizer.convert_ids_to_tokens(features['input_ids'][0])
        # find mask token index
        mask_id = torch.zeros(self.max_lenght)
        mask_pos = [i for i, x in enumerate(tokenized_text) if x == '<mask>']
        mask_id[mask_pos] = 1
        # example id
        ex_id = example['ex_id']
        return (features, mask_id, ex_id)
    
    def data_prompt_loader(self, examples, y, batch_size, shuffle=False):
        ''' return a tensor.data.loader with examples processed by prompt_tokenizer'''
        # tokenize examples
        input_ids = []
        attention_mask = []
        mask_ids = []
        ex_ids = []
        for example in examples:
            features, mask_id, ex_id = self.prompt_tokenizer(example)
            input_ids.append(features['input_ids'][0])
            attention_mask.append(features['attention_mask'][0])
            mask_ids.append(mask_id)
            ex_ids.append(ex_id)
        # join output
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        mask_ids = torch.stack(mask_ids)
        ex_ids = torch.tensor(ex_ids)
        # create dataset
        tensor_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, mask_ids, ex_ids, y)
        # create dataloader
        tensor_dataloader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)
        return tensor_dataloader
    
    def train_loop(self, train_loader, val_loader, epochs, loss_fn, optimizer, scheduler, val_epoch=1):
        # logs for tensorboard
        writer = SummaryWriter(os.path.join(self.save_dir, 'tb_logs', self.model_name))
        model = model.to(self.device)
        best_val_loss = 1000000
        best_epoch = 0
        with tqdm(range(epochs), desc=self.model_name) as tepoch:
            for epoch in tepoch:
                train_true = []
                train_pred = []
                train_loss = 0
                val_true = []
                val_pred = []
                # train model
                model.train()
                for input_ids, attention_mask, mask_ids, ex_ids, target in train_loader:
                    input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                    mask_ids = mask_ids.to(self.device)
                    target = target.to(self.device)
                    pred, loss_batch = self.train_model(model, input_ids, attention_mask, mask_ids, target, optimizer, loss_fn)
                    train_loss += loss_batch
                    train_true.append(target.cpu())
                    train_pred.append(pred)
                scheduler.step()
                # compute f1 score for positive class        
                f1_train = f1_score(torch.cat(train_true, dim=0), torch.cat(train_pred, dim=0), average='micro')
                # val model
                if (epoch % val_epoch) == 0:
                    val_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for input_ids, attention_mask, mask_ids, ex_ids, target in val_loader:
                            input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                            mask_ids = mask_ids.to(self.device)
                            target = target.to(self.device)
                            pred, loss_batch = self.eval_model(model, input_ids, attention_mask, mask_ids, target, loss_fn)
                            val_loss += loss_batch
                            val_true.append(target.cpu())
                            val_pred.append(pred)
                    # compute f1 score for positive class     
                    f1_val = f1_score(torch.cat(val_true, dim=0), torch.cat(val_pred, dim=0), average='micro') 

                # store weights based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.adapters:
                        # save only clf head and adapters
                        model.plm.save_adapter(os.path.join(self.save_dir, self.model_name, 'adapters'), 'adapter')
                        torch.save(model.clf_layer.state_dict(), os.path.join(self.save_dir, self.model_name, 'clf_layer.pth'))
                    else:
                        # save all model
                        torch.save(model.clf_layer.state_dict(), os.path.join(self.save_dir, self.model_name, 'clf_layer.pth'))
                        torch.save(model.plm.state_dict(), os.path.join(self.save_dir, self.model_name, 'encoder.pth'))
                    best_epoch = epoch
                
                tepoch.set_postfix(loss_train=round(train_loss/self.len_train, 5), loss_val=round(val_loss/self.len_val, 5), 
                                f1_train=f1_train, f1_val=f1_val, best_epoch=best_epoch)
                
                # logger
                writer.add_scalar('Loss/train', train_loss/self.len_train, epoch)
                writer.add_scalar('Loss/val', val_loss/self.len_val, epoch)
                writer.add_scalar('F1/train', f1_train, epoch)
                writer.add_scalar('F1/val', f1_val, epoch)