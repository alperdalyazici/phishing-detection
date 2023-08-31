import itertools

embedding_output_dim = [8, 16, 32]
lstm_units = [16, 32, 64]
epochs = [3]
batch_size = [16, 32, 64]


hyperparameter_configs = []

#Create and append all possible hyperparameter configurations into the list
for emb_dim in embedding_output_dim:
    for lstm_unit in lstm_units:
        for epoch in epochs:
            for batch_size_val in batch_size:
                config = {
                    'embedding_output_dim': emb_dim,
                    'lstm_units': lstm_unit,
                    'epochs': epoch,
                    'batch_size': batch_size_val
                }
                hyperparameter_configs.append(config)

