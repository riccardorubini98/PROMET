import json
import numpy as np
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
from sklearn.preprocessing import MultiLabelBinarizer


def import_json_examples(file_name):
    """ import json data """
    example = []
    with open(file_name, 'r') as f:
        for line in f:
            example.append(json.loads(line))
        f.close()
    return np.array(example)

def examples_preprocessing(examples):
    """ preprocess example
    
    return: 
        - a python dictionary: {'text': full sentence, 'mention', 'mention_type', 'ex_id}
    """
    clean_example = []
    for id, ex in enumerate(examples):
        left_string = TreebankWordDetokenizer().detokenize(ex['left_context_token'])
        right_string = TreebankWordDetokenizer().detokenize(ex['right_context_token'])
        full_string = left_string + ' ' + ex['mention_span'] + ' ' + right_string
        clean_example.append({'text': full_string, 'mention':ex['mention_span'], 'mention_type':ex['y_str'], 'ex_id':id})
    return np.array(clean_example)

def import_examples(file_name):
    """ import and preprocess examples 
    
    return:
        - np.array with preprocessed examples
        - np.array with mention types
    """
    examples_raw = import_json_examples(file_name)
    examples = examples_preprocessing(examples_raw)
    y =  np.array([[ex['mention_type'][-1]] for ex in examples])
    return examples, y

def prompt_tokenizer(example, template, tokenizer, max_lenght=128):
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
    prompt = template.format(mention=example['mention'])
    # generate example
    text = example['text'].strip('.') + '. ' + prompt
    # tokenize
    features = tokenizer.encode_plus(text, return_tensors="pt", truncation=True, 
                                     max_length=max_lenght, padding='max_length')
    tokenized_text = tokenizer.convert_ids_to_tokens(features['input_ids'][0])
    # find mask token index
    mask_id = torch.zeros(max_lenght)
    mask_pos = [i for i, x in enumerate(tokenized_text) if x == '<mask>']
    mask_id[mask_pos] = 1
    # example id
    ex_id = example['ex_id']
    return (features, mask_id, ex_id)

def data_prompt_loader(examples, y, template, tokenizer, max_lenght=128, batch_size=8, shuffle=False):
    ''' return a tensor.data.loader with examples processed by prompt_tokenizer'''
    # tokenize examples
    input_ids = []
    attention_mask = []
    mask_ids = []
    ex_ids = []
    for example in examples:
        features, mask_id, ex_id = prompt_tokenizer(example, template, tokenizer, max_lenght)
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

def add_other_class(y):
    # add /other vector (all zeros at the beginning)
    y = np.c_[y, np.zeros(y.shape[0])]
    # if no specialization (i.e. all zeros in a row) -> inference /type/other
    for y_row in y:
        if np.count_nonzero(y_row) == 0:
            y_row[-1] = 1
    return torch.tensor(y)

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
        y_train_e = add_other_class(mlb.fit_transform(y_train))
        y_val_e = add_other_class(mlb.transform(y_val))
        y_test_e = add_other_class(mlb.transform(y_test))
        # other class in last position
        labels_names = list(mlb.classes_) + [other_class]
        labels_names = [[name] for name in labels_names]
    else:
        mlb = MultiLabelBinarizer()
        y_train_e = torch.tensor(mlb.fit_transform(y_train))
        y_val_e = torch.tensor(mlb.transform(y_val))
        y_test_e = torch.tensor(mlb.transform(y_test))
        labels_names = list(mlb.classes_)
        labels_names = [[name] for name in labels_names]
    
    return (y_train_e, y_val_e, y_test_e), np.array(labels_names)