import torch
import torch.nn as nn
from transformers import BertModel, BartConfig, BartForSequenceClassification
from transformers.models.bart.modeling_bart import BartEncoder, BartPretrainedModel

# BERT
class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        
        x_atten_masks[:,0] = 0 # [CLS] --> 0 
        idx = torch.arange(0, last_hidden[0].shape[1], 1).to('cuda')
        x_seg_ind = x_seg_ids * idx
        x_att_ind = (x_atten_masks-x_seg_ids) * idx
        indices_seg = torch.argmax(x_seg_ind, 1, keepdim=True)
        indices_att = torch.argmax(x_att_ind, 1, keepdim=True)
        for seg, seg_id, att, att_id in zip(x_seg_ids, indices_seg, x_atten_masks, indices_att):
            seg[seg_id] = 0  # 2nd [SEP] --> 0 
            att[att_id:] = 0  # 1st [SEP] --> 0 
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_seg_ids.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_seg_ids.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out

    
# BART
class Encoder(BartPretrainedModel):
    
    def __init__(self, config: BartConfig):
        
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        self.encoder = BartEncoder(config, self.shared)

    def forward(self, input_ids, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return encoder_outputs

class bart_classifier(nn.Module):

    def __init__(self, num_labels, model_select, gen, dropout, dropoutrest):

        super(bart_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout) if gen==0 else nn.Dropout(dropoutrest)
        self.relu = nn.ReLU()
        
        self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
        self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
        self.bart.pooler = None
        self.linear = nn.Linear(self.bart.config.hidden_size*2, self.bart.config.hidden_size)
        self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
        last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
        eos_token_ind = x_input_ids.eq(self.config.eos_token_id).nonzero() # tensor([[0,4],[0,5],[0,11],[1,2],[1,3],[1,6]...])
        
        # print("x_input_ids:",x_input_ids,x_input_ids.size())
        # print("x_atten_masks:",x_atten_masks,x_atten_masks.size())

        # print("len(eos_token_ind):",len(eos_token_ind))
        # print("len(x_input_ids):",len(x_input_ids),3*len(x_input_ids))
        # print(bk)
        assert len(eos_token_ind) == 3*len(kwargs['input_ids'])
        b_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if i%3==0]
        e_eos = [eos_token_ind[i][1] for i in range(len(eos_token_ind)) if (i+1)%3==0]
        x_atten_clone = x_atten_masks.clone().detach()
        for begin, end, att, att2 in zip(b_eos, e_eos, x_atten_masks, x_atten_clone):
            att[begin:], att2[:begin+2] = 0, 0 # att all </s> --> 0; att2 1st and 2nd </s> --> 0
            att[0], att2[end] = 0, 0 # <s> --> 0; 3rd </s> --> 0
        
        txt_l = x_atten_masks.sum(1).to('cuda')
        topic_l = x_atten_clone.sum(1).to('cuda')
        txt_vec = x_atten_masks.type(torch.FloatTensor).to('cuda')
        topic_vec = x_atten_clone.type(torch.FloatTensor).to('cuda')
        txt_mean = torch.einsum('blh,bl->bh', last_hidden[0], txt_vec) / txt_l.unsqueeze(1)
        topic_mean = torch.einsum('blh,bl->bh', last_hidden[0], topic_vec) / topic_l.unsqueeze(1)
        
        cat = torch.cat((txt_mean, topic_mean), dim=1)
        query = self.dropout(cat)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out
    

# class bart_classifier(nn.Module):

#     def __init__(self, num_labels, model_select, gen, dropout):

#         super(bart_classifier, self).__init__()
        
#         self.dropout = nn.Dropout(0.1) if gen==0 else nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
#         self.config = BartConfig.from_pretrained('facebook/bart-large-mnli')
#         self.bart = Encoder.from_pretrained("facebook/bart-large-mnli")
#         self.linear = nn.Linear(self.bart.config.hidden_size, self.bart.config.hidden_size)
#         self.out = nn.Linear(self.bart.config.hidden_size, num_labels)
        
#     def forward(self, **kwargs):
        
#         x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']
#         last_hidden = self.bart(input_ids=x_input_ids, attention_mask=x_atten_masks)
        
#         hidden_states = last_hidden[0] 
#         eos_mask = x_input_ids.eq(self.config.eos_token_id)

#         if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
#             raise ValueError("All examples must have the same number of <eos> tokens.")
#         query = hidden_states[eos_mask,:].view(hidden_states.size(0), -1, hidden_states.size(-1))[:,-1,:]

#         query = self.dropout(query)
#         linear = self.relu(self.linear(query))
#         out = self.out(linear)
        
#         return out