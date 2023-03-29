import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model.components.encoder import Encoder
from model.components.decoder import Decoder
from typing import Union, Callable
import os
from model.utils.mask import generate_mask, generate_padding_mask
from .metric import WER

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


class WhisperModel(nn.Module):
    def __init__(self, token_size: int, n_mels: int,  n: int, embedding_dim: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]) -> None:
        super().__init__()
        self.encoder = Encoder(n=n, n_mels=n_mels, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.decoder = Decoder(token_size=token_size, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.classifer = nn.Linear(in_features=embedding_dim, out_features=token_size)
        self.to(device)
    def forward(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor):
        # padding_mask = generate_padding_mask(torch.mean(encoder_inputs, dim=-1))
        padding_mask = generate_padding_mask(decoder_inputs)
        look_ahead_mask = generate_mask(decoder_inputs)

        encoder_output = self.encoder(encoder_inputs, None)
        decoder_output = self.decoder(decoder_inputs, encoder_output, None, look_ahead_mask)
        output = self.classifer(decoder_output)
        return output
    
class Whisper:
    def __init__(self, 
                token_size: int,
                n_mels: int,
                n: int = 4, 
                embedding_dim: int = 384, 
                heads: int = 6, 
                d_ff: int = 2048,
                dropout_rate: float = 0.1, 
                eps: float = 0.1, 
                activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                learning_rate: float = 0.00003,
                checkpoint: str = None) -> None:
        self.model = WhisperModel(token_size=token_size, n_mels=n_mels, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, activation=activation)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.epoch = 0
        self.loss = 0.0
        self.checkpoint = checkpoint

        self.metric = WER()

        self.accuracy = 0.0

        if self.checkpoint is not None:
            self.__load_model(self.checkpoint)

    def __load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

    def load_model(self, path: str = None):
        if path is not None:
            self.checkpoint = path
            self.__load_model(path)
        elif self.checkpoint is not None:
            self.__load_model(self.checkpoint)
        

    def __save_model(self, path: str):

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }, path)

        print(f"Model is saved at {path}")

    def save_model(self, path: str = None):
        if path is not None:
            self.__save_model(path)
            self.checkpoint = path
        elif self.checkpoint is not None:
            self.__save_model(self.checkpoint)



    def build_dataset(self, inputs: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool):
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader
    
    def calculate_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = labels.size(0)

        loss = 0.0

        for batch in range(batch_size):
            loss += self.criterion(outputs[batch], labels[batch])
        
        loss = loss/batch_size

        return loss
    
    def train_step(self, encoder_inputs: torch.Tensor, decoder_inputs: torch.Tensor, labels: torch.Tensor) -> None:
        outputs = self.model(encoder_inputs, decoder_inputs)

        loss = self.calculate_loss(outputs, labels)

        loss.backward()
        self.optimizer.step()

        self.loss += loss.item()

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 1, batch_size: int = 1, shuffle: bool = True, mini_batch: int = 1):
        dataloader = self.build_dataset(X_train, y_train, batch_size, shuffle)

        total = len(dataloader)
        delta = total - (total//mini_batch)*mini_batch
        for _ in range(epochs):
            for index, data in enumerate(dataloader):
                encoder_inputs = data[0].to(device)
                decoder_inputs = data[1][:, :-1].to(device)
                labels = data[1][:, 1:].to(device)

                self.train_step(encoder_inputs, decoder_inputs, labels)

                if index%mini_batch == mini_batch-1:
                    print(f"Epoch {self.epoch+1} Batch: {index+1} Loss: {(self.loss/mini_batch):.4f}")
                    self.loss = 0.0
                elif index == total - 1:
                    print(f"Epoch {self.epoch+1} Batch {index+1} Loss: {(self.loss/delta):.4f}")
                    self.loss = 0.0
            self.epoch += 1

        print("Finished Training")

        if self.checkpoint is not None:
            self.__save_model(self.checkpoint)

    def predict(self, audio: torch.Tensor, decoder_in: torch.Tensor, limit_token: int, end_token: int):
        for _ in range(limit_token):
            output = self.model(audio, decoder_in)

            _, token = torch.max(output, dim=-1)

            if token == end_token:
                break

            decoder_in = torch.cat([decoder_in, token.unsequeeze(0)], dim=-1)

        return decoder_in
