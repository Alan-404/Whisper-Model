# Whisper Model
## I. Architecture


## II. Setup Environment
1. Make sure you have installed Python.
2. Python version `3.10.9`
3. cd `{project_folder}`
4. Install needed packages: `pip install requirements.txt`

## III. Parameters
- `vocab_size`: Number of tokens in Tokenizer.
- `sample_rate`: Number of samples get in 1 second of audio data.
- `duration`: Max time of audio data.
- `frame_size`: Length of the windowed signal after padding with zeros
- `hop_length`: Number of audio samples between adjacent STFT columns.
- `length_seq`: Max token in a sequence.
- `n_e`: Number of Encoder layers.
- `n_d`: Number of Decoder layers.
- `embedding_dim`: Dimension of Word2Vec.
- `heads`: Number of heads in Multi-head Attention.
- `d_ff`: Hidden neutrons in Position Wise Feed-Forward Networks.
- `dropout_rate`: Probability of an element to be zeroed.
- `eps`: A value added to the denominator for numerical stability.
- `activation`: Activation function in hidden layer.
- `m`: Number of 2-D Attention Layers.
- `channels`: Number of hidden channels in 2-D attention.