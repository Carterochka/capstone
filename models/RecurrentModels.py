# This file contains all the recurrent models that I am using for training
import numpy as np
import matplotlib.pyplot as plt

import sys

if '..' not in sys.path:
    sys.path.append('..')

import torch
from torch import nn

import torch.nn.functional as F

import abc

import reservoirpy as rpy
from reservoirpy.nodes import Input, Reservoir, Ridge, ReLU

def zero_one_constrain(tensor):
    ceil = torch.minimum(tensor, torch.ones(tensor.size()))
    return F.relu(ceil)

class RNNBaseClass:
    def run(self, *args, **kwargs):
        return zero_one_constrain(self(*args, **kwargs))

    @staticmethod
    def calculate_loss(model, loss_fn, dataloader, dataset, visualize_first_10_trajectories=True):
        loss = (np.sum([loss_fn(model.run(X), y)*len(y) for X, y in dataloader]) / len(dataset)) ** 0.5
        print('Calculated loss: ', loss)

        if visualize_first_10_trajectories:
            fig1, axs1 = plt.subplots(2, 5, figsize=(20,10))
            fig2, axs2 = plt.subplots(2, 5, figsize=(20,10))

            fig1.suptitle('Y-coordinate over time')
            fig2.suptitle('Trajectory')

            plt.setp(axs1[-1, :], xlabel='Frame number')
            plt.setp(axs1[:, 0], ylabel='Y-coordinate')

            plt.setp(axs2[-1, :], xlabel='X-coordinate')
            plt.setp(axs2[:, 0], ylabel='Y-coordinate')

            for row_id in range(len(axs1)):
                for col_id in range(len(axs1[row_id])):
                    # axs1 and axs2 are time series, so only limiting y-axis
                    axs1[row_id, col_id].set_ylim(0, 1)

                    # axs3 shows trajectories, so limiting both axis
                    axs2[row_id, col_id].set_xlim(0, 1)
                    axs2[row_id, col_id].set_ylim(0, 1)


            for X, y in dataloader:
                pred = model.run(X).detach().numpy()

                for count in range(10):
                    gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])
                    pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])

                    axs1[int(count >= 5)][count % 5].plot(gr[1::2], label=f'Ground truth')
                    axs1[int(count >= 5)][count % 5].plot(pr[1::2], label='Predicted')
                    axs1[int(count >= 5)][count % 5].legend()

                    axs2[int(count >= 5)][count % 5].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                    axs2[int(count >= 5)][count % 5].plot(pr[0::2], pr[1::2], label='Predicted')
                    axs2[int(count >= 5)][count % 5].legend()

                break

            plt.show()
        return loss
    

    @classmethod
    def train_model(model_class, train_dataloader, num_epochs=100, error_threshold=15, epoch_threshold=0.001, learning_rate=0.1, **kwargs):
        model = model_class(**kwargs)
        print(model)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # This array will store the loss values per epoch
        loss_per_epoch = []

        # This variable is needed to calculate the total epoch loss
        epoch_loss = 0

        # Training
        epoch = 0
        while epoch < num_epochs:
            for X, y in train_dataloader:
                # For each data instance we need to reset the optimizer gradients
                optimizer.zero_grad()
                
                pred = model(X)
                loss = loss_fn(pred, y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                
                
            if epoch_loss > error_threshold:
                print(f'Unsuccessful start. Loss: {epoch_loss}')
                model = model_class(**kwargs)
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
                epoch_loss = 0
                continue

            loss_per_epoch.append(epoch_loss)

            
            # Keeping training log
            print(f'Epoch {epoch} complete. Training loss: {epoch_loss}')

            if len(loss_per_epoch) > 1 and loss_per_epoch[-2] - epoch_loss < epoch_threshold:
                print(f'Training stopped on epoch {epoch}')
                break
            
            epoch_loss = 0
            epoch += 1

        return model


class VanilaRNN(nn.Module, RNNBaseClass):
    def __init__(self, input_dim, hidden_dim, num_rnns, output_dim, dropout_prob=0, relu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rnns = num_rnns

        self.rnn = nn.RNN(input_dim, hidden_dim, num_rnns, batch_first=True, dropout=dropout_prob)
        self.out = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.normal_(self.out.weight, 0, 1)
        self.relu = torch.nn.ReLU() if relu else None

    
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_rnns, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.out(out)

        if self.relu:
            out = self.relu(out)

        return out


class GRU(nn.Module, RNNBaseClass):
    def __init__(self, input_dim, hidden_dim, num_rnns, output_dim, dropout_prob=0, relu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rnns = num_rnns

        self.gru = nn.GRU(input_dim, hidden_dim, num_rnns, batch_first=True, dropout=dropout_prob)
        self.out = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.normal_(self.out.weight, 0, 1)
        self.relu = torch.nn.ReLU() if relu else None

    
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_rnns, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.gru(x, h0.detach())

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.out(out)

        if self.relu:
            out = self.relu(out)

        return out


class LSTM(nn.Module, RNNBaseClass):
    def __init__(self, input_dim, hidden_dim, num_rnns, output_dim, dropout_prob=0, relu=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_rnns = num_rnns

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_rnns, batch_first=True, dropout=dropout_prob)
        self.out = nn.Linear(hidden_dim, output_dim)
        torch.nn.init.normal_(self.out.weight, 0, 1)
        self.relu = torch.nn.ReLU() if relu else None

    
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.num_rnns, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.num_rnns, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.out(out)

        if self.relu:
            out = self.relu(out)

        return out


# This is the base class for the echo state models. It implements the loss calculation given the reservoir model specifics
class EchoBaseClass(abc.ABC):
    @abc.abstractmethod
    def __init__(self, input_dim, reservoir_size, output_dim, leaking_rate, spectral_radius, ridge_param):
        pass


    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)


    def run(self, *args, **kwargs):
        return zero_one_constrain(torch.tensor(self.model.run(*args, **kwargs)))


    @staticmethod
    def calculate_loss(model, loss_fn, dataloader, dataset, visualize_first_10_trajectories=True):
        loss = (np.sum([loss_fn(model.run(X.squeeze().numpy()), y.squeeze())*len(y) for X, y in dataloader]) / len(dataset)) ** 0.5
        print('Test loss: ', loss)

        if visualize_first_10_trajectories:
            fig1, axs1 = plt.subplots(2, 5, figsize=(20,10))
            fig2, axs2 = plt.subplots(2, 5, figsize=(20,10))

            fig1.suptitle('Y-coordinate over time')
            fig2.suptitle('Trajectory')

            plt.setp(axs1[-1, :], xlabel='Frame number')
            plt.setp(axs1[:, 0], ylabel='Y-coordinate')

            plt.setp(axs2[-1, :], xlabel='X-coordinate')
            plt.setp(axs2[:, 0], ylabel='Y-coordinate')

            for row_id in range(len(axs1)):
                for col_id in range(len(axs1[row_id])):
                    # axs1 and axs2 are time series, so only limiting y-axis
                    axs1[row_id, col_id].set_ylim(0, 1)

                    # axs3 shows trajectories, so limiting both axis
                    axs2[row_id, col_id].set_xlim(0, 1)
                    axs2[row_id, col_id].set_ylim(0, 1)


            for X, y in dataloader:
                pred = model.run(X.squeeze().numpy())

                for count in range(10):
                    gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])
                    pr = np.insert(pred.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])

                    axs1[int(count >= 5)][count % 5].plot(gr[1::2], label=f'Ground truth')
                    axs1[int(count >= 5)][count % 5].plot(pr[1::2], label='Predicted')
                    axs1[int(count >= 5)][count % 5].legend()

                    axs2[int(count >= 5)][count % 5].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                    axs2[int(count >= 5)][count % 5].plot(pr[0::2], pr[1::2], label='Predicted')
                    axs2[int(count >= 5)][count % 5].legend()

                break

            plt.show()
        return loss
    
    @classmethod
    def train_model(model_class, train_dataloader, **kwargs):
        network = model_class(**kwargs)
        print(network.model)

        for X, y in train_dataloader:
            network.model.fit(X.squeeze().numpy(), y.squeeze().numpy())

        return network


class ESN(EchoBaseClass):
    def __init__(self, input_dim, reservoir_size, output_dim, leaking_rate, spectral_radius, ridge_param, relu=False):
        reservoir = Reservoir(units=reservoir_size, lr=leaking_rate, sr=spectral_radius, input_bias=False)
        readout = Ridge(output_dim=output_dim, ridge=ridge_param)

        self.model = reservoir >> readout

        if relu:
            self.model = self.model >> ReLU()


class SeqESN(EchoBaseClass):
    def __init__(self, input_dim, reservoir_size, output_dim, leaking_rate, spectral_radius, ridge_param, number_of_reservoirs=1, relu=False):
        self.model = Input()

        for _ in range(number_of_reservoirs):
            reservoir = Reservoir(units=reservoir_size, lr=leaking_rate, sr=spectral_radius, input_bias=False)
            readout = Ridge(output_dim=output_dim, ridge=ridge_param)
            self.model = self.model >> reservoir >> readout

        if relu:
            self.model = self.model >> ReLU()


class ParallelESN(EchoBaseClass):
    def __init__(self, input_dim, reservoir_size, output_dim, leaking_rate, spectral_radius, ridge_param, number_of_reservoirs=1, relu=False):
        self.model = []
        for _ in range(number_of_reservoirs):
            reservoir = Reservoir(units=reservoir_size, lr=leaking_rate, sr=spectral_radius, input_bias=False)
            self.model.append(reservoir)

        parallels = self.model[0]

        for i in range(1, number_of_reservoirs):
            parallels >> self.model[i]

        readout = Ridge(output_dim=output_dim, ridge=ridge_param)# >> ReLU()
        self.model.append(parallels)
        self.model = self.model >> readout

        if relu:
            self.model = self.model >> ReLU()


class GroupedESN(EchoBaseClass):
    def __init__(self, input_dim, reservoir_size, output_dim, leaking_rate, spectral_radius, ridge_param, number_of_reservoirs=1, relu=False):
        self.model = []
        for _ in range(number_of_reservoirs):
            reservoir = Reservoir(units=reservoir_size, lr=leaking_rate, sr=spectral_radius, input_bias=False)
            self.model.append(reservoir)

        readout = Ridge(output_dim=output_dim, ridge=ridge_param)# >> ReLU()
        self.model = self.model >> readout

        if relu:
            self.model = self.model >> ReLU()