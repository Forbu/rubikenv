import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import rubikenv.rubikgym as rb
#from rubikenv.utils import estimate_solvability_rate_searchOptim

class RubikTransformer_search(pl.LightningModule):
    """
    RubikTransformer model for search among state node
    """

    def __init__(self, hidden_size=256, num_layers=4, num_heads=4, dropout=0.1, color_embedding_size=5):
        """
        Initialize the model
        :param input_size: input size of the model
        :param hidden_size: hidden size of the model
        :param num_layers: number of layers of the model
        :param num_heads: number of heads of the model
        :param dropout: dropout probability
        :param device: device to run the model on
        """
        super(RubikTransformer_search, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.color_embedding_size = color_embedding_size

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=128, batch_first=True), num_layers
        )
        
        # color embedding
        self.color_embedding = nn.Embedding(6, color_embedding_size)

        # first layer of the model
        self.first_layer = nn.Linear(3*3*6*color_embedding_size, hidden_size)


        # last layer to get the value
        self.value_layer = nn.Linear(hidden_size, 1)

        self.loss_value = nn.MSELoss()


    def forward(self, state):
        """
        state : Tensor of dim (batch, nb_state, 3*3*6) with only int
        """
        batch_size, nb_state, nb_face = state.shape
        
        # get the device of state tensor
        device = state.device

        # generate the color embedding to get a dimension of (batch, nb_state, nb_face, color_embedding_size)
        color_embedding = self.color_embedding(state)

        # reshape the color embedding to get a dimension of (batch, nb_state, nb_face*color_embedding_size)
        color_embedding = color_embedding.view(batch_size, nb_state, nb_face*self.color_embedding_size)

        # now we concatenate the spatial and color embedding to get a dimension of (batch, 9, 6, spatial_embedding_size + color_embedding_size)
        embedding = color_embedding

        # apply the first layer
        embedding = self.first_layer(embedding)
        embedding = torch.relu(embedding)

        # we compute the mask to remove the padding
        attn_mask = torch.tril(torch.ones((batch_size, nb_state)))

        # send attn_mask to the device of state tensor
        attn_mask = attn_mask.to(device)

        # we apply the encoder to the embedding to get a dimension of (batch, 9*6, hidden_size)
        embedding = self.encoder(embedding)

        # we get the value
        value = self.value_layer(embedding)

        return value # shape of the input is (batch, nb_state, 1)

    def training_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward = batch
        value = self.forward(state.long())

        loss = self.loss_value(value.float(), reward.float())

        self.log('train_loss_value', loss, prog_bar=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """
        At the end of the training epoch we check the average number of randomize rubik's cube solved
        using estimate_solvability_rate_searchOptim
        """
        pass

    def validation_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward = batch
        value = self.forward(state.long())
        loss = self.loss_value(value.float(), reward.float())/10

        self.log('val_loss_value', loss, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        """
        return the optimizer and the scheduler
        :return:
        """
        return torch.optim.AdamW(self.parameters(), lr=0.001)