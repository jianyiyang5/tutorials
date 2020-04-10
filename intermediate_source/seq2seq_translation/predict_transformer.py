import torch
from model_transformer import TransformerMT, generate_square_subsequent_mask
from train_transformer import load_model
from data import MAX_LENGTH, prepareData, tensor_with_mask, UNK_token, zeroPadding, PAD_token, batch_to_transformer_data


def target_tensor(tgt_sentences, tgt_voc):
    indexes_batch = [[tgt_voc.word2index[word] if word in tgt_voc.word2index else UNK_token
                      for word in tgt_sentence.split(' ')] for tgt_sentence in tgt_sentences]
    padList = zeroPadding(indexes_batch)
    mask_matrix = []
    for pad in padList:
        mask = [True if idx == PAD_token else False for idx in pad]
        mask_matrix.append(mask)
    padVar = torch.LongTensor(padList)
    mask_tensor = torch.BoolTensor(mask_matrix)
    return padVar, mask_tensor



def predict(src_voc, tgt_voc, src_sentences, model, device):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        tgt_sentences = ['SOS'] * len(src_sentences)
        for _ in range(MAX_LENGTH):
            src_tensor, src_pad_mask = tensor_with_mask(src_sentences, src_voc)
            tgt_tensor, tgt_pad_mask = target_tensor(tgt_sentences, tgt_voc)
            mem_pad_mask = src_pad_mask.clone()
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)
            src_pad_mask = src_pad_mask.to(device)
            # tgt_pad_mask = tgt_pad_mask.to(device)
            mem_pad_mask = mem_pad_mask.to(device)
            tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0)).to(device)
            outputs = model(src_tensor, tgt_tensor, src_key_padding_mask=None,
                                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                    tgt_mask=tgt_mask)
            for i in range(outputs.size(1)):
                _, indices = torch.topk(outputs[-1][i], 5)
                tgt_sentences[i] = tgt_sentences[i] + ' ' + tgt_voc.index2word[indices[0].item()]
    return tgt_sentences


def evaluate(src_voc, tgt_voc, pairs, model, device):
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        src_tensor, tgt_tensor, src_pad_mask, tgt_pad_mask, mem_pad_mask = \
            batch_to_transformer_data(src_voc, tgt_voc, pairs)
        src_tensor = src_tensor.to(device)
        tgt_tensor = tgt_tensor.to(device)
        src_pad_mask = src_pad_mask.to(device)
        tgt_pad_mask = tgt_pad_mask.to(device)
        mem_pad_mask = mem_pad_mask.to(device)
        tgt_mask = generate_square_subsequent_mask(tgt_tensor.size(0) - 1).to(device)
        outputs = model(src_tensor, tgt_tensor[:-1, :], src_key_padding_mask=src_pad_mask,
                        tgt_key_padding_mask=tgt_pad_mask[:-1, :], memory_key_padding_mask=mem_pad_mask,
                        tgt_mask=tgt_mask)
        predicted_list = []
        for i in range(outputs.size(1)):
            output = outputs[:, i, :]
            predicted = []
            for j in range(output.size(0)):
                _, indices = torch.topk(output[j], 1)
                predicted.append(tgt_voc.index2word[indices[0].item()])
            predicted_list.append(' '.join(predicted))

    return predicted_list


if __name__ == '__main__':
    model, _ = load_model('output/transformer.pt.9')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_voc, tgt_voc, pairs = prepareData('eng', 'fra', True, '../data')
    src_sentences = [src for src, _ in pairs[100000:100008]]
    tgt_sentences = [tgt for _, tgt in pairs[100000:100008]]
    predicted = predict(src_voc, tgt_voc, src_sentences, model, device)
    for sents in zip(predicted, tgt_sentences):
        print(f'hyp: {sents[0]}')
        print(f'ref: {sents[1]}\n')
    # evaluate(src_voc, tgt_voc, pairs[:8], model, device)


