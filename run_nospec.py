from data import data_prompt_loader, import_examples
from model import Promet, promet_clf
from utils import one_hot_encoder
from torch import nn
import torch
import os

class PrometNoSpec(Promet):
    
    def train_model(model, input_ids, attention_mask, mask_ids, y, optimizer, loss_fn):
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, mask_ids)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        posterior = torch.softmax(output, dim=-1)
        pred = torch.argmax(posterior, dim=-1)
        return pred.cpu(), loss.item()

    def eval_model(model, input_ids, attention_mask, mask_ids, y, loss_fn):
        output = model(input_ids, attention_mask, mask_ids)
        loss = loss_fn(output, y)
        posterior = torch.softmax(output, dim=-1)
        pred = torch.argmax(posterior, dim=-1)
        return pred.cpu(), loss.item()

    def fit(plm, train_examples, val_examples, y_train, y_val, lr, lambd, wd, epochs, 
                batch_size, n_class, val_epoch=1, adapters=False):
        
        self.n_class = n_class
        self.len_train = len(train_examples)
        self.len_val = len(val_examples)
        self.adapters = adapters
        # loader
        train_loader = self.data_prompt_loader(train_examples, y_train, batch_sizeshuffle=True)
        val_loader = self.data_prompt_loader(val_examples, y_val, batch_size, shuffle=False)
        # loss
        loss_fn = nn.CrossEntropyLoss()
        # init model
        model = promet_clf(self.plm_name, self.n_class, lambd=lambd)
        
        if type(adapters)==bool and adapters==True:
            # create new_adapters
            adapter_name = model.plm.add_adapter('adapter', config='pfeiffer')
            model.plm.active_adapters = adapter_name
            # freeze layer
            for name, p in model.plm.named_parameters():
                if 'adapters' not in name:
                    p.requires_grad=False
        elif type(adapters) == str:
            # load pre_trained adapters
            adapter_name = model.plm.load_adapter(adapters)
            model.plm.active_adapters = adapter_name
            # freeze layer
            for name, p in model.plm.named_parameters():
                if 'adapters' not in name:
                    p.requires_grad=False
        else:
            pass
        
        # optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer_plm= list(model.plm.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer_plm
                        if not any(nd in n for nd in no_decay)], 'lr':lr, 'weight_decay': 1e-2, 'correct_bias':True},
            {'params': [p for n, p in param_optimizer_plm
                        if any(nd in n for nd in no_decay)], 'lr':lr, 'weight_decay': 0, 'correct_bias':True},
            {'params': [model.clf_layer.weight], 'lr':lr, 'weight_decay': wd, 'correct_bias':True},
            {'params': [model.clf_layer.bias], 'lr':lr, 'weight_decay': 0, 'correct_bias':True}
                        ]
        optim = torch.optim.AdamW(optimizer_grouped_parameters)
        
        # lr schedule
        num_training_steps = epochs * self.len_train / batch_size
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        
        # create dir to save model
        if not os.path.exists(os.path.join(self.save_dir, self.model_name)):
            os.makedirs(os.path.join(self.save_dir, self.model_name))
            
        # train model
        self.train_loop(model, train_loader, val_loader, optim, 
                scheduler, loss_fn, epochs=epochs , val_epoch=val_epoch)
    
    def return_model(self):
        model = promet_clf(self.plm_name, self.n_class)
        if self.adapters:
            # load only adapters and clf_head
            model.clf_layer.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name, 'clf_layer.pth')))
            adapter_name = model.plm.load_adapter(os.path.join(self.save_dir, self.model_name, 'adapters'))
            model.plm.active_adapters = adapter_name
        else:
            model.plm.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name, 'encoder.pth')))
            model.clf_layer.load_state_dict(torch.load(os.path.join(self.save_dir, self.model_name, 'clf_layer.pth')))
            