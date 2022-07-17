import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import rubikenv.rubikgym as rb
from rubikenv.utils import estimate_solvability_rate

class RubikTransformer(pl.LightningModule):
    """
    RubikTransformer model
    """

    def __init__(self, hidden_size=128, num_layers=4, num_heads=4, dropout=0.1, spatial_embedding_size=64, color_embedding_size=64, output_size=12):
        """
        Initialize the model
        :param input_size: input size of the model
        :param hidden_size: hidden size of the model
        :param num_layers: number of layers of the model
        :param num_heads: number of heads of the model
        :param dropout: dropout probability
        :param device: device to run the model on
        """
        super(RubikTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.spatial_embedding_size = spatial_embedding_size
        self.color_embedding_size = color_embedding_size
        self.output_size = output_size

        assert hidden_size == color_embedding_size + spatial_embedding_size, "hidden_size and color_embedding_size + spatial_embedding_size must be equal"

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dim_feedforward=64, batch_first=True), num_layers
        )
        

        # spatial parameters of dim (1, 9, spatial_embedding_size)
        self.spatial_embedding = nn.parameter.Parameter(torch.randn(1, 9, 6, spatial_embedding_size))
        
        # color embedding
        self.color_embedding = nn.Embedding(6, color_embedding_size)

        # last layer to get the action probabilities
        self.last_layer = nn.Linear(hidden_size, output_size)

        # last layer to get the value
        self.value_layer = nn.Linear(hidden_size, 1)

        # loss function logits and labels and value
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.loss_value = nn.MSELoss()


    def forward(self, state):
        """
        state : Tensor of dim (batch, 9, 6) with only int
        """
        batch_size, nb_spatial, nb_color = state.shape

        # generate the spatial embedding to get a dimension of (batch, 9, 6, spatial_embedding_size)
        spatial_embedding = self.spatial_embedding.repeat(batch_size, 1, 1, 1)

        # generate the color embedding to get a dimension of (batch, 9, 6, color_embedding_size)
        color_embedding = self.color_embedding(state)

        # now we concatenate the spatial and color embedding to get a dimension of (batch, 9, 6, spatial_embedding_size + color_embedding_size)
        embedding = torch.cat((spatial_embedding, color_embedding), dim=3)

        # reshape embedding from (batch, 9, 6, spatial_embedding_size + color_embedding_size) to (batch, 9*6, spatial_embedding_size + color_embedding_size)
        embedding = embedding.view(batch_size, 9 * 6, -1)

        # we apply the encoder to the embedding to get a dimension of (batch, 9*6, hidden_size)
        embedding = self.encoder(embedding)

        # now we compute the mean of the hidden size to get a dimension of (batch, hidden_size)
        embedding = embedding.mean(dim=1)

        # now we can apply the last layer of the model
        action_logits = self.last_layer(embedding)

        # we get the value
        value = self.value_layer(embedding)

        return action_logits, value

    def training_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward, reverse_action = batch
        action_logits, value = self.forward(state.long())

        loss_value = self.loss_value(value.float(), reward.float())
        loss_action = self.loss(action_logits, reverse_action.long()) 
        loss = loss_action + loss_value

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_value', loss_value, prog_bar=True)
        self.log('train_action', loss_action, prog_bar=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """
        At the end of the training epoch we check the average number of randomize rubik's cube solved
        using estimate_solvability_rate
        """
        avg_solvability = estimate_solvability_rate(model=self, nb_try=20, batch_size=12)
        self.log('avg_solvability', avg_solvability)

    
    def validation_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward, reverse_action = batch
        action_logits, value = self.forward(state.long())
        loss_value = self.loss_value(value.float(), reward.float())/10
        loss_action = self.loss(action_logits, reverse_action.long()) 
        loss = loss_action + loss_value

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_value', loss_value, prog_bar=True)
        self.log('val_action', loss_action, prog_bar=True)

        return {'loss': loss}



    def configure_optimizers(self):
        """
        return the optimizer and the scheduler
        :return:
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)

class RubikDense(pl.LightningModule):
    """
    RubikDense model
    MLP with dense layers
    """

    def __init__(self, hidden_size=1024, color_embedding_size=5, output_size=12):
        """
        Initialize the model
        :param input_size: input size of the model
        :param hidden_size: hidden size of the model
        :param num_layers: number of layers of the model
        :param num_heads: number of heads of the model
        :param dropout: dropout probability
        :param device: device to run the model on
        """
        super(RubikDense, self).__init__()
        self.input_size = 9*6* color_embedding_size
        self.hidden_size = hidden_size
        self.color_embedding_size = color_embedding_size
        self.output_size = output_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # color embedding
        self.color_embedding = nn.Embedding(6, color_embedding_size)

        # last layer to get the action probabilities
        self.last_layer = nn.Linear(hidden_size, output_size)

        # last layer to get the value
        self.value_layer = nn.Linear(hidden_size, 1)

        # loss function logits and labels and value
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.loss_value = nn.MSELoss()


    def forward(self, state):
        """
        state : Tensor of dim (batch, 9, 6) with only int
        """
        batch_size, nb_spatial, nb_color = state.shape

        # reshape state into (batch, 9*6)
        state = state.view(batch_size, 9 * 6)

        # generate the color embedding to get a dimension of (batch, 9*6, color_embedding_size)
        color_embedding = self.color_embedding(state)

        # now we concatenate the spatial and color embedding to get a dimension of (batch, 9, spatial_embedding_size + color_embedding_size)
        embedding = color_embedding

        # flatten the embedding to get a dimension of (batch, 9*6* color_embedding_size)
        embedding = embedding.view(batch_size, -1)

        # now we can apply the encoder to the embedding to get a dimension of (batch, 9 * hidden_size)
        embedding = self.encoder(embedding)

        # now we can apply the last layer of the model
        action_logits = self.last_layer(embedding)

        # we get the value
        value = self.value_layer(embedding)

        return action_logits, value
    
    def training_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward, reverse_action = batch
        action_logits, value = self.forward(state.long())

        loss_value = self.loss_value(value.float(), reward.float())
        loss_action = self.loss(action_logits, reverse_action.long()) 
        loss = loss_action + loss_value

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_value', loss_value, prog_bar=True)
        self.log('train_action', loss_action, prog_bar=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        """
        At the end of the training epoch we check the average number of randomize rubik's cube solved
        using estimate_solvability_rate
        """
        avg_solvability = estimate_solvability_rate(model=self, nb_try=20, batch_size=12)
        self.log('avg_solvability', avg_solvability)

    
    def validation_step(self, batch, batch_idx):
        """
        load the data from batch and run the forward pass
        :param batch:
        :param batch_idx:
        :return:
        """
        state, reward, reverse_action = batch
        action_logits, value = self.forward(state.long())
        loss_value = self.loss_value(value.float(), reward.float())/10
        loss_action = self.loss(action_logits, reverse_action.long()) 
        loss = loss_action + loss_value

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_value', loss_value, prog_bar=True)
        self.log('val_action', loss_action, prog_bar=True)

        return {'loss': loss}

    def configure_optimizers(self):
        """
        return the optimizer and the scheduler
        :return:
        """
        return torch.optim.Adam(self.parameters(), lr=0.001)
