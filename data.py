import pandas as pd
from argparse import ArgumentParser
from preprocessing.audio import AudioProcessor
from preprocessing.text import TextProcessor
import numpy as np
import pickle


parser = ArgumentParser()

parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--duration', type=int, default=10)
parser.add_argument("--frame_size", type=int, default=int(16000*0.025))
parser.add_argument("--hop_length", type=int, default=int(16000*0.01))
parser.add_argument("--n_mels", type=int, default=80)
parser.add_argument("--length_seq", type=int, default=46)

parser.add_argument("--raw_data", type=str)
parser.add_argument("--audio_folder", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--clean_data", type=str)

args = parser.parse_args()


def save_clean_data(data: np.ndarray, path: str):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def program(sample_rate: int, duration: float, frame_size: int, hop_length: int, n_mels: int, length_seq: int, tokenizer_path: str, csv_path: str, audio_folder: str, clean_path: str):
    audio_processor = AudioProcessor(sample_rate=sample_rate, duration=duration, mono=True, frame_size=frame_size, hop_length=hop_length, n_mels=n_mels)
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    df = pd.read_csv(csv_path)
    
    file_names = df['filename'].values.tolist()
    contents = df['content'].values.tolist()

    audio_data  = audio_processor.process(folder_path=audio_folder, list_names=file_names)
    text_data = text_processor.process(data=contents, max_len=length_seq, start_token=True, end_token=True)


    save_clean_data(audio_data, f"{clean_path}/audio.pkl")
    save_clean_data(text_data, f"{clean_path}/text.pkl")

if __name__ == "__main__":
    print(args.raw_data)
    if args.raw_data is None or args.audio_folder is None or args.tokenizer is None or args.clean_data is None:
        print("Missing data")
    else:
        program(
            sample_rate=args.sample_rate,
            duration=args.duration,
            frame_size=args.frame_size,
            hop_length=args.hop_length,
            length_seq=args.length_seq,
            tokenizer_path=args.tokenizer,
            csv_path=args.raw_data,
            audio_folder=args.audio_folder,
            clean_path=args.clean_data,
            n_mels = args.n_mels
        )