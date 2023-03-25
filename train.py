import torch
from argparse import ArgumentParser
from typing import Union, Callable
from model.whisper import Whisper
from preprocessing.text import TextProcessor
parser = ArgumentParser()

parser.add_argument("--n", type=int, default=4)
parser.add_argument("--embedding_dim", type=int, default=384)
parser.add_argument('--heads', type=int, default=6)
parser.add_argument("--d_ff", type=int, default=2048)
parser.add_argument("--dropout_rate", type=float, default=0.1)
parser.add_argument("--eps", type=float, default=0.1)
parser.add_argument("--activation", type=str, default='relu')

parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--mini_batch", type=int, default=1)
parser.add_argument("--shuffle", type=bool, default=True)
parser.add_argument("--learning_rate", type=float, default=0.0006)

parser.add_argument("--tokenizer", type=str)
parser.add_argument("--data_audio", type=str)
parser.add_argument("--data_text", type=str)
parser.add_argument("--checkpoint", type=str)

args = parser.parse_args()

def program(n: int, 
            embedding_dim: int,
            heads: int, 
            d_ff: int,  
            dropout_rate: float, 
            eps: float, 
            activation: Union[str, Callable[[torch.Tensor], torch.Tensor]],
            tokenizer_path: str,
            path_audio: str,
            path_text: str,
            learning_rate: float,
            checkpoint: str,
            epochs: int,
            batch_size: int,
            mini_batch: int,
            shuffle: bool):
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)
    audio_data = text_processor.load_data(path_audio)
    text_data = text_processor.load_data(path_text)

    text_processor.loadd_tokenizer(tokenizer_path)

    token_size = text_processor.tokenizer.num_tokens + 1
    n_mels = audio_data.shape[-1]

    model = Whisper(token_size=token_size, n_mels=n_mels, n=n, embedding_dim=embedding_dim, heads=heads, d_ff=d_ff, dropout_rate=dropout_rate, eps=eps, learning_rate=learning_rate, checkpoint=checkpoint)

    audio_data = torch.tensor(audio_data)
    text_data = torch.tensor(text_data)

    model.fit(X_train=audio_data, y_train=text_data, epochs=epochs, batch_size=batch_size, shuffle=shuffle, mini_batch=mini_batch)


if __name__ == '__main__':
    if args.checkpoint is None or args.tokenizer is None or args.data_audio is None or args.data_text is None:
        print("Missing Information")
    else:
        program(
            n=args.n,
            embedding_dim=args.embedding_dim,
            heads=args.heads,
            d_ff=args.d_ff,
            dropout_rate=args.dropout_rate,
            eps=args.eps,
            activation=args.activation,
            tokenizer_path=args.tokenizer,
            path_audio=args.data_audio,
            path_text=args.data_text,
            learning_rate=args.learning_rate,
            checkpoint = args.checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            mini_batch=args.mini_batch,
            shuffle=args.shuffle
        )