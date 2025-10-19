import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset, Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
import json

class Dataset(Dataset):
    def __init__(self, df, seqlen=5000, smilen=600):
        self.proteins = df['proteins'].values
        self.ligands = df['ligands'].values
        self.affinity = df['affinity'].values
        self.seqlen = seqlen
        self.smilelen = smilen
        self.protein_vocab = set()
        self.ligand_vocab = set()
        for pr in self.proteins:
            for i in pr:
                self.protein_vocab.update(i)
        for lig in self.ligands:
            for i in lig:
                self.ligand_vocab.update(i)
        self.protein_vocab.update(['dummy'])
        self.ligand_vocab.update(['dummy'])
        self.protein_dict = {x: i for i, x in enumerate(self.protein_vocab)}
        self.ligand_dict = {x: i for i, x in enumerate(self.ligand_vocab)}

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        pr = self.proteins[idx]
        lig = self.ligands[idx]
        target = self.affinity[idx]
        protein = [self.protein_dict[x] for x in pr] + [self.protein_dict['dummy']] * (self.seqlen - len(pr))  
        ligand = [self.ligand_dict[x] for x in lig] + [self.ligand_dict['dummy']] * (self.smilelen - len(lig))
        
        return torch.tensor(protein), torch.tensor(ligand), torch.tensor(target, dtype=torch.float)

class Conv1d(nn.Module):
    def __init__(self, vocab_size, channel, kernel_size, stride=1, padding=0):
        super(Conv1d, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=128)
        # 蛋白质序列的每个字母被嵌入为128个特征的向量 每个字母的第一个特征一起表示该蛋白质的第一个特征
        self.conv1 = nn.Conv1d(128, channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv1d(channel, channel*2, kernel_size, stride, padding)
        self.conv3 = nn.Conv1d(channel*2, channel*3, kernel_size, stride, padding)
        # self.conv4 = nn.Conv1d(channel*3, channel*4, kernel_size, stride, padding)
        # self.conv5 = nn.Conv1d(channel*4, channel*3, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.globalmaxpool = nn.AdaptiveMaxPool1d(1)
    
    def forward(self, x):
        x = self.embedding(x) #embedding层处理后维度为(batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1) #维度变换 conv1需要(batch_size, channels, sequence_length)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.conv5(x)
        x = self.relu(x)
        x = self.globalmaxpool(x)
        x = x.squeeze(-1)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.w_o(attn_output)
        return output

class DeepDTA(nn.Module):
    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTA, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        self.fc1 = nn.Linear(channel*6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein, ligand):
        x1 = self.ligand_conv(ligand)
        x2 = self.protein_conv(protein)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x.squeeze()

class DeepDTAWithAttention(nn.Module):
    def __init__(self, pro_vocab_size, lig_vocab_size, channel, protein_kernel_size, ligand_kernel_size):
        super(DeepDTAWithAttention, self).__init__()
        self.ligand_conv = Conv1d(lig_vocab_size, channel, ligand_kernel_size)
        self.protein_conv = Conv1d(pro_vocab_size, channel, protein_kernel_size)
        
        self.cross_attention = MultiHeadAttention(d_model=channel*3, num_heads=8)
        
        self.fc1 = nn.Linear(channel*6, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, protein, ligand):
        ligand_features = self.ligand_conv(ligand)  # (batch, channel*3)
        protein_features = self.protein_conv(protein)  # (batch, channel*3)
        
        # 添加序列维度用于注意力计算
        ligand_seq = ligand_features.unsqueeze(1)  # (batch, 1, channel*3)
        protein_seq = protein_features.unsqueeze(1)  # (batch, 1, channel*3)
        
        # 交叉注意力：药物查询蛋白质，蛋白质查询药物
        ligand_attended = self.cross_attention(ligand_seq, protein_seq, protein_seq)  # (batch, 1, channel*3)
        protein_attended = self.cross_attention(protein_seq, ligand_seq, ligand_seq)  # (batch, 1, channel*3)
        
        # 移除序列维度并拼接
        ligand_attended = ligand_attended.squeeze(1)
        protein_attended = protein_attended.squeeze(1)
        
        x = torch.cat((ligand_attended, protein_attended), dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x.squeeze()

def collate_fn(batch):
    """
    Collate function for the DataLoader
    """
    proteins, ligands, targets = zip(*batch)
    proteins = torch.stack(proteins, dim=0)
    ligands = torch.stack(ligands, dim=0)
    targets = torch.stack(targets, dim=0)
    return proteins, ligands, targets

class Trainer:
    def __init__(self, model, channel, protein_kernel, ligand_kernel, df, train_idx, val_idx, test_idx,
                 log_file, seqlen=5000, smilen=600):
        self.dataset = Dataset(df, smilen=smilen, seqlen=seqlen)
        self.protein_vocab = len(self.dataset.protein_vocab) + 1
        self.ligand_vocab = len(self.dataset.ligand_vocab) + 1
        self.train_dataset = Subset(self.dataset, train_idx)
        self.val_dataset = Subset(self.dataset, val_idx)
        self.test_dataset = Subset(self.dataset, test_idx)
        self.protein_kernel = protein_kernel
        self.ligand_kernel = ligand_kernel
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model(self.protein_vocab, self.ligand_vocab, channel, protein_kernel, ligand_kernel).to(self.device)
        
        #日志
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def train(self, lr, num_epochs, batch_size, save_path):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        writer = SummaryWriter()
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, num_workers=16, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last = False, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=16, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last = False, collate_fn=collate_fn)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=16, prefetch_factor=2, pin_memory=True, persistent_workers=True, drop_last = False, collate_fn=collate_fn)

        with open('./json/protein_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.protein_dict, f)
        with open('./json/ligand_dict-prk{}-ldk{}.json'.format(self.protein_kernel, self.ligand_kernel), 'w') as f:
            json.dump(self.dataset.ligand_dict, f)

        best_weights = self.model.state_dict()
        best_val_loss = np.inf
        best_epoch = 0

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            with tqdm(total=len(train_loader)) as pbar:
                for protein, ligand, target in train_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                    pbar.update(1)

            train_loss /= len(train_loader)
            self.logger.info('Epoch: {} - Training Loss: {:.6f}'.format(epoch+1, train_loss))
            writer.add_scalar('train_loss', train_loss, epoch)

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for protein, ligand, target in val_loader:
                    protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                    output = self.model(protein, ligand)
                    loss = criterion(output, target)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = deepcopy(self.model.state_dict())
                best_epoch = epoch
                self.logger.info('Best Model So Far in Epoch: {}'.format(epoch+1))
            self.logger.info('Epoch: {} - Validation Loss: {:.6f}'.format(epoch+1, val_loss))
            writer.add_scalar('val_loss', val_loss, epoch)
        
        self.model.load_state_dict(best_weights)
        test_result = []
        with torch.no_grad():
            for protein, ligand, target in test_loader:
                protein, ligand, target = protein.to(self.device), ligand.to(self.device), target.to(self.device)

                output = self.model(protein, ligand)
                test_result.append(output.cpu().numpy())
        test_result = np.concatenate(test_result)
        np.savetxt('./TestLog/test-result-prk{}-ldk{}.txt'.format(self.protein_kernel, self.ligand_kernel), test_result)
        
        self.logger.info('Best Model Loaded from Epoch: {}'.format(best_epoch+1))
        torch.save(self.model.state_dict(), save_path)
        self.logger.handlers[0].close()
        self.logger.removeHandler(self.logger.handlers[0])
        writer.close()

fp = 'kiba_all_with_split.csv'
df = pd.read_csv(fp)
train_idx = df[df['split'] == 'train'].index.values
val_idx = df[df['split'] == 'val'].index.values
test_idx = df[df['split'] == 'test'].index.values

model = DeepDTAWithAttention
channel = 32
# protein_kernel = [8, 12]
# ligand_kernel = [4, 8]
protein_kernel = [12]
ligand_kernel = [8]
if __name__ == '__main__':
    for prk in protein_kernel:
        for ldk in ligand_kernel:
            trainer = Trainer(model, channel, prk, ldk, df, train_idx, val_idx, test_idx, "./TrainingLog/training-prk{}-ldk{}.log".format(prk, ldk))
            trainer.train(num_epochs=150, batch_size=512, lr=0.001, save_path='./ReTrainingPt/deepdta_retrain-prk{}-ldk{}.pt'.format(prk, ldk))
