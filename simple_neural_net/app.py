from src.create_layers import create_layers, get_data
from src.linear_model import linear_model

train_df, test_df = get_data()
preprocessing_layers, inputs = create_layers(train_df)
linear_model(preprocessing_layers, train_df, test_df, inputs, 0.1, 10, 50)
