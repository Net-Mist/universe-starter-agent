from model import *

possible_model = ['LSTM', 'VIN', 'FF', 'VINBigger']
one_input_brain = ['LSTM']  # List of the brain that use only one input frame. The rest of the input are RNN data

model_name_to_class = {
    'LSTM': LSTMPolicy,
    'VIN': VINPolicy,
    'VINBigger': VINBiggerPolicy,
    'FF': FFPolicy
}
