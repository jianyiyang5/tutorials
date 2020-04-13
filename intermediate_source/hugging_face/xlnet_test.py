from transformers import XLNetTokenizer, XLNetModel
import torch

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetModel.from_pretrained('xlnet-large-cased')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=False)).unsqueeze(0)  # Batch size 1

outputs = model(input_ids)
last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(last_hidden_states[0].size())


# from transformers import XLNetTokenizer, XLNetLMHeadModel
# import torch
#
# tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
# model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

# # We show how to setup inputs to predict a next token using a bi-directional context.
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
# perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
# perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
# target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
# target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)
#
# outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
# next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
# print(next_token_logits.size())
# predicted_index = torch.argmax(next_token_logits[0, -1, :])
# print(predicted_index)
# print(tokenizer.decode([predicted_index.item()]))


from transformers import XLNetTokenizer, XLNetLMHeadModel
import torch

tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
model = XLNetLMHeadModel.from_pretrained('xlnet-large-cased')

# We show how to setup inputs to predict a next token using a bi-directional context.
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping)
next_token_logits = outputs[0]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]

# The same way can the XLNetLMHeadModel be used to be trained by standard auto-regressive language modeling.
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is very <mask>", add_special_tokens=False)).unsqueeze(0)  # We will predict the masked token
labels = torch.tensor(tokenizer.encode("cute", add_special_tokens=False)).unsqueeze(0)
assert labels.shape[0] == 1, 'only one word will be predicted'
perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float)
perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token as is done in standard auto-regressive lm training
target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float)  # Shape [1, 1, seq_length] => let's predict one token
target_mapping[0, 0, -1] = 1.0  # Our first (and only) prediction will be the last token of the sequence (the masked token)

outputs = model(input_ids, perm_mask=perm_mask, target_mapping=target_mapping, labels=labels)
loss, next_token_logits = outputs[:2]  # Output has shape [target_mapping.size(0), target_mapping.size(1), config.vocab_size]
print(next_token_logits.size())
predicted_index = torch.argmax(next_token_logits[0, -1, :])
print(predicted_index)
print(tokenizer.decode([predicted_index.item()]))