from data import data_prompt_loader, import_examples, one_hot_encoder
from model import Promet, promet_clf, set_seed
from torch import nn
import torch
import os
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from score import get_score
import yaml
import json

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PrometNoSpec(Promet):
    
    def train_model(self, model, input_ids, attention_mask, mask_ids, y, optimizer, loss_fn):
        optimizer.zero_grad()
        output = model(input_ids, attention_mask, mask_ids)
        y = y.argmax(dim=1) #one_hot -> single label
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        posterior = torch.softmax(output, dim=-1)
        pred_label = posterior.argmax(dim=1)
        # prediction in one-hot-encoding
        pred = torch.zeros_like(posterior).scatter_(1, pred_label.unsqueeze(1), 1.)
        return pred.cpu(), loss.item()

    def eval_model(self, model, input_ids, attention_mask, mask_ids, y, loss_fn):
        output = model(input_ids, attention_mask, mask_ids)
        y = y.argmax(dim=1) #one_hot -> single label
        loss = loss_fn(output, y)
        posterior = torch.softmax(output, dim=-1)
        pred_label = posterior.argmax(dim=1)
        # prediction in one-hot-encoding
        pred = torch.zeros_like(posterior).scatter_(1, pred_label.unsqueeze(1), 1.)
        return pred.cpu(), loss.item()

    def fit(self, train_examples, val_examples, y_train, y_val, lr, lambd, wd, epochs, 
                batch_size, n_class, val_epoch=1, adapters=False):
        
        self.n_class = n_class
        self.len_train = len(train_examples)
        self.len_val = len(val_examples)
        self.adapters = adapters
        self.batch_size = batch_size
        # loader
        train_loader = self.data_prompt_loader(train_examples, y_train, batch_size, shuffle=True)
        val_loader = self.data_prompt_loader(val_examples, y_val, batch_size, shuffle=False)
        # loss
        loss_fn = nn.CrossEntropyLoss()
        # init model
        model = promet_clf(self.plm_name, self.n_class, lambd=lambd)
        # add adapters if needed
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
        n_train_par = count_trainable_parameters(model)
        print(f'Trainable parameters: {n_train_par}')
        # optimizer
        optim = self.optimzer_promet(model, lr, wd)
        # lr schedule
        num_training_steps = epochs * self.len_train / batch_size
        num_warmup_steps = int(0.1 * num_training_steps)
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # create dir to save model
        if not os.path.exists(os.path.join(self.save_dir, self.model_name)):
            os.makedirs(os.path.join(self.save_dir, self.model_name))
        # train model
        self.train_loop(model, train_loader, val_loader, epochs, loss_fn, optim, 
                scheduler, val_epoch=val_epoch)
        # return best model on validation
        self.model = self.return_model()
    
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
        return model
    
    def predict_logits(self, examples, verbose=True):
        # loader
        data_loader = self.data_prompt_loader(examples, torch.zeros(len(examples)), self.batch_size, shuffle=False)
        self.model.to(self.device)
        self.model.eval()
        posterior = []
        with torch.no_grad():
            for input_ids, attention_mask, mask_ids, ex_ids, target in tqdm(data_loader, disable=not verbose):
                input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
                mask_ids = mask_ids.to(self.device)
                posterior.append(self.model(input_ids, attention_mask, mask_ids))
        posterior = torch.cat(posterior)
        return posterior.cpu()
    
    def predict(self, examples, verbose=True):
        logits = self.predict_logits(examples, verbose=verbose)
        posterior = torch.softmax(logits, dim=-1)
        pred_label = posterior.argmax(dim=1)
        # prediction in one-hot-encoding
        pred = torch.zeros_like(posterior).scatter_(1, pred_label.unsqueeze(1), 1.)
        return pred


if __name__ == "__main__":
    
    # PARAMTERS
    with open('config_nospec.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    PLM_NAME = config['MODEL']['PLM_NAME']
    TRAIN_FILE = config['DATA']['TRAIN_FILE']
    VAL_FILE = config['DATA']['VAL_FILE']
    TEST_FILE = config['DATA']['TEST_FILE']
    MASK_TOKEN = config['MODEL']['MASK_TOKEN']
    SAVE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), config['DATA']['SAVE_DIR'])
    ADAPTERS = config['MODEL']['ADAPTERS']
    # Hp
    BATCH_SIZE = config['TRAIN']['BATCH_SIZE']
    LR = float(config['TRAIN']['LR'])
    LAMBD = float(config['TRAIN']['LAMBD'])
    WD = float(config['TRAIN']['WD'])
    EPOCHS = config['TRAIN']['EPOCHS']
    SEED = config['TRAIN']['SEED']
    N_MASK = config['TRAIN']['N_MASK']
    
    # reproduciblity
    set_seed(SEED)
    
    # import data
    print('Load Data...')
    train_examples, y_train = import_examples(TRAIN_FILE)
    val_examples, y_val = import_examples(VAL_FILE)
    test_examples, y_test = import_examples(TEST_FILE)
    print('Data loaded')
    
    # format target
    (y_train, y_val, y_test), labels_names = one_hot_encoder(y_train, y_val, y_test)
    # add fathers names
    labels_names = np.array([['/' + label[0].split('/')[1], label[0]] for label in labels_names])
    
    # define model
    promet_nospec = PrometNoSpec(plm_name=PLM_NAME, n_mask=N_MASK, mask_token=MASK_TOKEN, 
                           model_name='no_spec', save_dir=SAVE_DIR)
    # fit mode
    print('Fit model')
    promet_nospec.fit(train_examples, val_examples, y_train, y_val, LR, LAMBD, WD, EPOCHS, BATCH_SIZE,
                      adapters=ADAPTERS, n_class=len(labels_names))
    
    # validation inference
    print('Validation inference:')
    y_val_hat = promet_nospec.predict(val_examples)
    acc, f1_micro, val_df = get_score(y_val, y_val_hat, labels_names)
    print(f'Validation -> Accuracy: {acc} \t F1-Score: {f1_micro}')
    # write on result file
    with open(os.path.join(SAVE_DIR, 'clf_results.txt'), 'a+') as f:
        f.write((json.dumps(config['DATA'])))
        f.write('\n')
        f.write(json.dumps(config['MODEL']))
        f.write('\n')
        f.write(f'Validation -> Accuracy: {acc} \t F1-Score: {f1_micro}')
        f.write('\n')
    
    # test inference
    print('Test inference:')
    y_test_hat = promet_nospec.predict(test_examples)
    acc, f1_micro, test_df = get_score(y_test, y_test_hat, labels_names)
    print(f'Test -> Accuracy: {acc} \t F1-Score: {f1_micro}')
    # write on result file
    with open(os.path.join(SAVE_DIR, 'clf_results.txt'), 'a+') as f:
        f.write(f'Test -> Accuracy: {acc} \t F1-Score: {f1_micro}')
        f.write('\n')
    # save metric report to csv
    test_df.to_csv(os.path.join(SAVE_DIR, TRAIN_FILE.split('/')[-1].replace('.json', '.csv')))