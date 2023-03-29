import re
import numpy as np
import pickle
import os

class SignHandler:
    def __init__(self) -> None:
        pass

    def sign(self, sequences: list, start_token: bool, end_token: bool) -> str:
        for index in range(len(sequences)):
            if start_token:
                sequences[index] = f"__start__ {sequences[index]}"
            if end_token:
                sequences[index] = f"{sequences[index]} __end__"
        return sequences

class Cleaner:
    def __init__(self, filters: str = r'[!@.,$#:)(][)]') -> None:
        self.filters = filters

    def __clean(self, sequence: str):
        sequence = sequence.lower()
        sequence = re.sub(r"[.,/:'\"\)\()]", "", sequence)
        sequence = re.sub(self.filters, "", sequence)
        sequence = sequence.strip()
        return sequence

    def clean(self, sequences: list):
        for i in range(len(sequences)):
            sequences[i] = self.__clean(sequences[i])

        return sequences
    
class Tokenizer:
    def __init__(self, tokenizer_path: str = None) -> None:
        self.tokenizer_path = tokenizer_path
        self.token_index = dict()
        self.index_token = dict()
        self.token_counts = dict()
        self.num_tokens = 0

    def __save_tokenizer(self, tokenizer_path: str = None) -> None:
        if tokenizer_path is not None:
            self.tokenizer_path = tokenizer_path
        with open(f'{self.tokenizer_path}', 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer is saved at {self.tokenizer_path}")

    def __load_tokenizer(self) -> None:
        if self.tokenizer_path is None:
            return
        if os.path.exists(f"{self.tokenizer_path}") == False:
            return
        with open(f'{self.tokenizer_path}', 'rb') as file:
            self = pickle.load(file)

    def add_token(self, token: str) -> None:
        if token not in self.token_index:
            self.num_tokens += 1
            self.token_index[token] = self.num_tokens
            self.index_token[self.num_tokens] = token
            self.token_counts[token] = 1
        else:
            self.token_counts[token] += 1

    def __tokenize(self, sequence: str) -> None:
        for text in sequence.split(' '):
            self.add_token(text)

    def get_index_token(self, index: int) -> str | None:
        if index > self.num_tokens:
            return
        return self.index_token[index]

    def get_token_index(self, token: str) -> int:
        if token not in self.token_index:
            self.add_token(token)
        return self.token_index[token]

    def __fit(self, sequence: str) -> np.ndarray:
        digit = list()
        for text in sequence.split(' '):
            digit.append(self.get_token_index(text))
        return np.array(digit)
    
    def tokenize(self, sequences: list) -> list:
        digit_sequences = []
        if self.tokenizer_path is not None:
            self.__load_tokenizer()
        for sequence in sequences:
            self.__tokenize(sequence=sequence)
            digit_sequences.append(self.__fit(sequence=sequence))
        if self.tokenizer_path is not None:
            self.__save_tokenizer(self.tokenizer_path)

        return digit_sequences

    def save_tokenizer(self, tokenizer_path: str) -> None:
        self.__save_tokenizer(tokenizer_path=tokenizer_path)
        self.tokenizer_path = tokenizer_path


class TextProcessor:
    def __init__(self, tokenizer_path: str) -> None:
        self.sign_handler =  SignHandler()
        self.cleanner = Cleaner()
        self.tokenizer = Tokenizer(tokenizer_path=tokenizer_path)

    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)
    
    def process(self, data: list, max_len: int = None, padding: str = 'post', truncating: str = "post", start_token: bool = False, end_token: bool = False) -> np.ndarray:
        sequences = self.sign_handler.sign(data, start_token=start_token, end_token=end_token)
        sequences = self.cleanner.clean(data)
        sequences = self.tokenizer.tokenize(sequences=sequences)
        if max_len is not None:
            sequences = self.pad_sequences(sequences=sequences, maxlen=max_len, padding=padding, truncating=truncating)
        return sequences

    def __load_data(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data

    def __save_data(self, data: np.ndarray, path: str, filename: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(f"{path}/{filename}", file, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_tokenizer(self, path: str) -> None:
        with open(path, 'rb') as file:
            self.tokenizer = pickle.load(file)

    def save_data(self, data: np.ndarray, path: str, filename: str, overwrite: bool = True) -> None:
        if os.path.exists(path) == True:
            if overwrite == True:
                self.__save_data(data, path, filename)
            else:
                print("Not Overwrite")
        else:
            print("Not found path")

    def load_data(self, path: str) -> np.ndarray | None:
        if os.path.exists(path) == True:
            data = self.__load_data(path)
            return data
        else:
            print("Not found path")

    def loadd_tokenizer(self, path: str) -> None:
        if os.path.exists(path) == True:
            self.__load_tokenizer(path)
            print("Loaded Tokenizer")
        else:
            print("Not found path")