import torch
from model_transformer import TransformerMT
from train_transformer import load_model


def predict(src_voc, tgt_voc, src_sentences, model, device):
    with torch.no_grad():
        model = model.to(device)
        model.eval()


