from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

def one_hot_encoder(y_train, y_val, y_test, other_class=''):
    """ formatting target in one-hot encoding
    
    return:
        - encoded target
        - labels_names: list with class name
    """
    if len(other_class)>0:
        # other_class encoded with [0,0,...,0]
        y_unique = np.unique(y_train)
        mlb = MultiLabelBinarizer(classes = [y for y in y_unique if y != other_class])
        # other class in last position
        labels_names = list(mlb.classes_) + [other_class]
    else:
        mlb = MultiLabelBinarizer()
        labels_names = list(mlb.classes_)
    mlb = MultiLabelBinarizer()
    y_train_e = mlb.fit_transform(y_train)
    y_val_e = mlb.transform(y_val)
    y_test_e = mlb.transform(y_test)
    
    return (y_train_e, y_val_e, y_test_e), labels_names

def train_loop(model, train_model, eval_model, train_loader, val_loader, dict_info, optimizer, 
               scheduler, loss_fn, epochs=30, verbose=True, val_epoch=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(dict_info['save_dir'], 'tb_logs', dict_info['model_name']))
    model = model.to(device)
    best_val_loss = 1000000
    best_epoch = 0
    with tqdm(range(epochs), desc=dict_info['model_name'], disable= not verbose) as tepoch:
        for epoch in tepoch:
            train_true = []
            train_pred = []
            train_loss = 0
            val_true = []
            val_pred = []
            # train model
            model.train()
            for input_ids, attention_mask, mask_ids, ex_ids, target in train_loader:
                input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                mask_ids = mask_ids.to(device)
                target = target.to(device)
                pred, loss_batch = train_model(model, input_ids, attention_mask, mask_ids, target, optimizer, loss_fn)
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
                        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                        mask_ids = mask_ids.to(device)
                        target = target.to(device)
                        pred, loss_batch = eval_model(model, input_ids, attention_mask, mask_ids, target, loss_fn)
                        val_loss += loss_batch
                        val_true.append(target.cpu())
                        val_pred.append(pred)
                # compute f1 score for positive class     
                f1_val = f1_score(torch.cat(val_true, dim=0), torch.cat(val_pred, dim=0), average='micro') 

            # store weights based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if dict_info['adapters']==True:
                    # save only clf head and adapters
                    model.plm.save_adapter(os.path.join(dict_info['save_dir'], dict_info['model_name'], 'adapters'), 'adapter')
                    torch.save(model.clf_layer.state_dict(), os.path.join(dict_info['save_dir'], dict_info['model_name'], 'clf_head.pth'))
                else:
                    # save all model
                    torch.save(model.clf_layer.state_dict(), os.path.join(dict_info['save_dir'], dict_info['model_name'], 'clf_head.pth'))
                    torch.save(model.plm.state_dict(), os.path.join(dict_info['save_dir'], dict_info['model_name'], 'encoder.pth'))
                best_epoch = epoch
            
            tepoch.set_postfix(loss_train=round(train_loss/dict_info['len_train'],5), loss_val=round(val_loss/dict_info['len_val'],5), 
                               f1_train=f1_train, f1_val=f1_val, best_epoch=best_epoch)
            
            # logger
            writer.add_scalar('Loss/train', train_loss/dict_info['len_train'], epoch)
            writer.add_scalar('Loss/val', val_loss/dict_info['len_val'], epoch)
            writer.add_scalar('F1/train', f1_train, epoch)
            writer.add_scalar('F1/val', f1_val, epoch)