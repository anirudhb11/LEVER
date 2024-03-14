import torch
import torch.nn as nn
import torch.nn.functional as F
import sentence_transformers

class STransformerInputLayer(nn.Module):
    """
    Sentence transformer
    """
    def __init__(self, transformer='roberta-base'):
        super(STransformerInputLayer, self).__init__()
        if isinstance(transformer, str):
            self.transformer = sentence_transformers.SentenceTransformer(transformer)
        else:
            self.transformer = transformer

    def forward(self, data):
        sentence_embedding = self.transformer(data)['sentence_embedding']
        return sentence_embedding

class CustomEncoder(nn.Module):
    """
    Encoder layer with Sentence transformer and an optional projection layer

    * projection layer is applied after reduction and normalization
    """
    def __init__(self, encoder_name, transform_dim):
        super(CustomEncoder, self).__init__()
        self.encoder = STransformerInputLayer(sentence_transformers.SentenceTransformer(encoder_name))
        self.transform_dim = transform_dim

    def forward(self, input_ids, attention_mask):
        return self.encoder({'input_ids': input_ids, 'attention_mask': attention_mask})

    @property
    def repr_dims(self):
        return  768

class SiameseNetwork(nn.Module):
    """
    A network class to support Siamese style training
    * specialized for sentence-bert or hugging face
    * hard-coded to use a joint encoder

    """
    def __init__(self, encoder_name, transform_dim, device, normalize_repr):
        super(SiameseNetwork, self).__init__()
        self.padding_idx = 0
        self.encoder = CustomEncoder(encoder_name, transform_dim)
        self.device = device
        self.normalize_repr = normalize_repr
        
    def encode(self, doc_input_ids, doc_attention_mask):
        rep = self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep

    def encode_document(self, doc_input_ids, doc_attention_mask, *args):
        rep = self.encoder(doc_input_ids.to(self.device), doc_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep

    def encode_label(self, lbl_input_ids, lbl_attention_mask):
        rep = self.encoder(lbl_input_ids.to(self.device), lbl_attention_mask.to(self.device))
        if self.normalize_repr:
            rep = F.normalize(rep)
        return rep
    
    def forward(self, doc_input_ids, doc_attention_mask, lbl_input_ids, lbl_attention_mask):
        if(doc_input_ids is None):
            return self.encode_label(lbl_input_ids, lbl_attention_mask)
        elif(lbl_input_ids is None):
            return self.encode_document(doc_input_ids, doc_attention_mask)
        doc_embeddings = self.encode_document(doc_input_ids, doc_attention_mask)
        label_embeddings = self.encode_label(lbl_input_ids, lbl_attention_mask)
        return doc_embeddings, label_embeddings

    @property
    def repr_dims(self):
        return self.encoder.repr_dims
    
class MyDataParallel(nn.DataParallel):
    """Allows data parallel to work with methods other than forward"""
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def prepare_network(args):
    """
    Set-up the network

    * Use DP if multiple GPUs are available
    """
    print("==> Creating model, optimizer...")
    snet = SiameseNetwork(args.encoder_name, args.transform_dim, args.device, True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        snet = MyDataParallel(snet)
    snet.to(args.device)

    print(snet)
    return snet

if __name__ == '__main__':
    snet = SiameseNetwork('msmarco-distilbert-base-v4', -1, 'cuda', True)