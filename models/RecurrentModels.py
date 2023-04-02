# This file contains all the recurrent models that I am using for training
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

if '..' not in sys.path:
    sys.path.append('..')

import torch
from torch import nn

import torch.nn.functional as F

import abc

import reservoirpy as rpy
from reservoirpy.nodes import Input, Reservoir, Ridge, ReLU

import pickle

# Helper function to constrain the outputs of the networks between 0 and 1.
def zero_one_constrain(tensor):
    ceil = torch.minimum(tensor, torch.ones(tensor.size()))
    return F.relu(ceil)

# Utility class for drawing the line with thickness from data space
# Source: https://stackoverflow.com/questions/19394505/expand-the-line-with-specified-width-in-data-unit/42972469#42972469

class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


#############################################
# Traditional RNN implementations
#############################################


class RNNBaseClass:
    def run(self, *args, **kwargs):
        return zero_one_constrain(self(*args, **kwargs))

    @staticmethod
    def calculate_loss(model, loss_fn, dataloader, dataset, visualize_first_10_trajectories=True, viz_shape=(2, 5)):
        loss = (np.sum([loss_fn(model.run(X), y)*len(y) for X, y in dataloader]) / len(dataset)) ** 0.5
        print('Calculated loss: ', loss)

        if visualize_first_10_trajectories:
            data_shape = dataset[0][0].shape[1]

            # Determining which scenario the data corresponds to from its shape
            # If the input shape is 2, it's a single ball 1D free-fall prediction
            if data_shape == 2:
                fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                fig1.suptitle('Y-coordinate over time')

                plt.setp(axs1[-1, :], xlabel='Frame number')
                plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                for row_id in range(len(axs1)):
                    for col_id in range(len(axs1[row_id])):
                        # axs1 and axs2 are time series, so only limiting y-axis
                        axs1[row_id, col_id].set_ylim(0, 1)

                for X, y in dataloader:
                    pred = model.run(X).detach().numpy()

                    for count in range(viz_shape[0]*viz_shape[1]):
                        gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0])
                        pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0])

                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                    break

                plt.show()
            
            # If the input shape is 3, it is the 2D single ball free-fall prediction
            elif data_shape == 3:
                fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                fig2, axs2 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                fig1.suptitle('Y-coordinate over time')
                fig2.suptitle('Trajectory')

                plt.setp(axs1[-1, :], xlabel='Frame number')
                plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                plt.setp(axs2[-1, :], xlabel='X-coordinate')
                plt.setp(axs2[:, 0], ylabel='Y-coordinate')

                for row_id in range(len(axs1)):
                    for col_id in range(len(axs1[row_id])):
                        # axs1 shows time series, so only limiting y-axis
                        axs1[row_id, col_id].set_ylim(0, 1)

                        # axs2 shows trajectories, so limiting both axis
                        axs2[row_id, col_id].set_xlim(0, 1)
                        axs2[row_id, col_id].set_ylim(0, 1)


                for X, y in dataloader:
                    pred = model.run(X).detach().numpy()

                    for count in range(viz_shape[0]*viz_shape[1]):
                        gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])
                        pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])

                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                        axs2[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                        axs2[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], pr[1::2], label='Predicted')
                        axs2[count // viz_shape[1]][count % viz_shape[1]].legend()

                    break

                plt.show()

            # If the shape is 9, it can be either three-ball prediction (free-fall or collisions) or entire scene prediction
            elif data_shape == 9:
                out_shape = dataset[0][1].shape[1]

                # If the output shape is 48 (X and Y coordinate of single ball on 24 frames), visualize a single ball prediction
                if out_shape == 48:
                    fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                    fig2, axs2 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                    fig3, axs3 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                    fig1.suptitle('Y-coordinate over time')
                    fig2.suptitle('X-coordinate over time')
                    fig3.suptitle('Trajectory')

                    plt.setp(axs1[-1, :], xlabel='Frame number')
                    plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                    plt.setp(axs2[-1, :], xlabel='Frame number')
                    plt.setp(axs2[:, 0], ylabel='X-coordinate')

                    plt.setp(axs3[-1, :], xlabel='X-coordinate')
                    plt.setp(axs3[:, 0], ylabel='Y-coordinate')

                    for row_id in range(len(axs1)):
                        for col_id in range(len(axs1[row_id])):
                            # axs1 and axs2 are time series, so only limiting y-axis
                            axs1[row_id, col_id].set_ylim(0, 1)
                            axs2[row_id, col_id].set_ylim(0, 1)

                            # axs3 shows trajectories, so limiting both axis
                            axs3[row_id, col_id].set_xlim(0, 1)
                            axs3[row_id, col_id].set_ylim(0, 1)


                    for X, y in dataloader:
                        pred = model.run(X).detach().numpy()

                        for count in range(viz_shape[0]*viz_shape[1]):
                            gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][-3:-1])
                            pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][-3:-1])

                            axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                            axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                            axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                            axs2[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], label=f'Ground truth')
                            axs2[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], label='Predicted')
                            axs2[count // viz_shape[1]][count % viz_shape[1]].legend()

                            axs3[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                            axs3[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], pr[1::2], label='Predicted')
                            axs3[count // viz_shape[1]][count % viz_shape[1]].legend()

                        break

                    plt.show()
            
                # If the output shape is 144 (X and Y coordinates of 3 balls on 24 frames), visualize the entire scene evolution
                elif out_shape == 144:
                    fig, axs = plt.subplots(5, 5, figsize=(20, 15))
                    
                    fig.suptitle('Trajectories of the balls', y=0.95)
                    
                    plt.setp(axs[-1, :], xlabel='X-coordinate')
                    plt.setp(axs[:, 0], ylabel='Y-coordinate')

                    for row_id in range(len(axs)):
                        for col_id in range(len(axs[row_id])):
                            axs[row_id, col_id].set_xlim(0, 1)
                            axs[row_id, col_id].set_ylim(0, 1)

                    col_names = ['Blue ball', 'Green ball', 'Red ball', 'Scene (ground truth)', 'Scene (predicted)']

                    for ax, col in zip(axs[0], col_names):
                        ax.set_title(col)


                    for X, y in dataloader:
                        pred = model.run(X).detach().numpy()

                        for count in tqdm(range(5), desc='Plotting progress'):
                            gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][[0, 1, 3, 4, 6, 7]])
                            pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][[0, 1, 3, 4, 6, 7]])

                            axs[count, 0].plot(gr[0::6], gr[1::6], color='blue', label='Ground truth')
                            axs[count, 0].plot(pr[0::6], pr[1::6], color='orange', label='Predicted')
                            axs[count, 0].legend()

                            axs[count, 1].plot(gr[2::6], gr[3::6], color='green', label='Ground truth')
                            axs[count, 1].plot(pr[2::6], pr[3::6], color='orange', label='Predicted')
                            axs[count, 1].legend()

                            axs[count, 2].plot(gr[4::6], gr[5::6], color='red', label='Ground truth')
                            axs[count, 2].plot(pr[4::6], pr[5::6], color='orange', label='Predicted')
                            axs[count, 2].legend()

                            data_linewidth_plot(gr[0::6], gr[1::6], ax=axs[count, 3], color='blue', linewidth=X.squeeze().numpy()[count][2], alpha=0.6)
                            data_linewidth_plot(gr[2::6], gr[3::6], ax=axs[count, 3], color='green', linewidth=X.squeeze().numpy()[count][5], alpha=0.6)
                            data_linewidth_plot(gr[4::6], gr[5::6], ax=axs[count, 3], color='red', linewidth=X.squeeze().numpy()[count][8], alpha=0.6)

                            data_linewidth_plot(pr[0::6], pr[1::6], ax=axs[count, 4], color='blue', linewidth=X.squeeze().numpy()[count][2], alpha=0.6)
                            data_linewidth_plot(pr[2::6], pr[3::6], ax=axs[count, 4], color='green', linewidth=X.squeeze().numpy()[count][5], alpha=0.6)
                            data_linewidth_plot(pr[4::6], pr[5::6], ax=axs[count, 4], color='red', linewidth=X.squeeze().numpy()[count][8], alpha=0.6)

                        break

                    plt.show()

                else:
                    print('Visual cannot be created as scenario cannot be identified from the data shape')
            else:
                print('Visual cannot be created as scenario cannot be identified from the data shape')
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

            # if len(loss_per_epoch) > 1 and loss_per_epoch[-2] - epoch_loss < epoch_threshold:
            #     print(f'Training stopped on epoch {epoch}')
            #     break

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









#############################################
# Echo State Network Implementations
#############################################


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
    def calculate_loss(model, loss_fn, dataloader, dataset, visualize_first_10_trajectories=True, viz_shape=(2, 5)):
        loss = (np.sum([loss_fn(model.run(X.squeeze().numpy()), y.squeeze())*len(y) for X, y in dataloader]) / len(dataset)) ** 0.5
        print('Test loss: ', loss)

        if visualize_first_10_trajectories:
            data_shape = dataset[0][0].shape[1]

            if data_shape == 2:
                fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                fig1.suptitle('Y-coordinate over time')

                plt.setp(axs1[-1, :], xlabel='Frame number')
                plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                for row_id in range(len(axs1)):
                    for col_id in range(len(axs1[row_id])):
                        # axs1 and axs2 are time series, so only limiting y-axis
                        axs1[row_id, col_id].set_ylim(0, 1)

                for X, y in dataloader:
                    pred = model.run(X.squeeze().numpy())

                    for count in range(viz_shape[0]*viz_shape[1]):
                        gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0])
                        pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0])

                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                    break

                plt.show()
            
            elif data_shape == 3:
                fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                fig2, axs2 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                fig1.suptitle('Y-coordinate over time')
                fig2.suptitle('Trajectory')

                plt.setp(axs1[-1, :], xlabel='Frame number')
                plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                plt.setp(axs2[-1, :], xlabel='X-coordinate')
                plt.setp(axs2[:, 0], ylabel='Y-coordinate')

                for row_id in range(len(axs1)):
                    for col_id in range(len(axs1[row_id])):
                        # axs1 shows time series, so only limiting y-axis
                        axs1[row_id, col_id].set_ylim(0, 1)

                        # axs2 shows trajectories, so limiting both axis
                        axs2[row_id, col_id].set_xlim(0, 1)
                        axs2[row_id, col_id].set_ylim(0, 1)


                for X, y in dataloader:
                    pred = model.run(X.squeeze().numpy())

                    for count in range(viz_shape[0]*viz_shape[1]):
                        gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])
                        pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][0:2])

                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                        axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                        axs2[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                        axs2[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], pr[1::2], label='Predicted')
                        axs2[count // viz_shape[1]][count % viz_shape[1]].legend()

                    break

                plt.show()

            elif data_shape == 9:
                out_shape = dataset[0][1].shape[1]

                # If the output shape is 2 (X and Y coordinate of single ball), visualize a single ball prediction
                if out_shape == 48:
                    fig1, axs1 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                    fig2, axs2 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))
                    fig3, axs3 = plt.subplots(viz_shape[0], viz_shape[1], figsize=(20,10))

                    fig1.suptitle('Y-coordinate over time')
                    fig2.suptitle('X-coordinate over time')
                    fig3.suptitle('Trajectory')

                    plt.setp(axs1[-1, :], xlabel='Frame number')
                    plt.setp(axs1[:, 0], ylabel='Y-coordinate')

                    plt.setp(axs2[-1, :], xlabel='Frame number')
                    plt.setp(axs2[:, 0], ylabel='X-coordinate')

                    plt.setp(axs3[-1, :], xlabel='X-coordinate')
                    plt.setp(axs3[:, 0], ylabel='Y-coordinate')

                    for row_id in range(len(axs1)):
                        for col_id in range(len(axs1[row_id])):
                            # axs1 and axs2 are time series, so only limiting y-axis
                            axs1[row_id, col_id].set_ylim(0, 1)
                            axs2[row_id, col_id].set_ylim(0, 1)

                            # axs3 shows trajectories, so limiting both axis
                            axs3[row_id, col_id].set_xlim(0, 1)
                            axs3[row_id, col_id].set_ylim(0, 1)


                    for X, y in dataloader:
                        pred = model.run(X.squeeze().numpy())

                        for count in range(viz_shape[0]*viz_shape[1]):
                            gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][-3:-1])
                            pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][-3:-1])

                            axs1[count // viz_shape[1]][count % viz_shape[1]].plot(gr[1::2], label=f'Ground truth')
                            axs1[count // viz_shape[1]][count % viz_shape[1]].plot(pr[1::2], label='Predicted')
                            axs1[count // viz_shape[1]][count % viz_shape[1]].legend()

                            axs2[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], label=f'Ground truth')
                            axs2[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], label='Predicted')
                            axs2[count // viz_shape[1]][count % viz_shape[1]].legend()

                            axs3[count // viz_shape[1]][count % viz_shape[1]].plot(gr[0::2], gr[1::2], label=f'Ground truth')
                            axs3[count // viz_shape[1]][count % viz_shape[1]].plot(pr[0::2], pr[1::2], label='Predicted')
                            axs3[count // viz_shape[1]][count % viz_shape[1]].legend()

                        break

                    plt.show()
                
                elif out_shape == 144:
                    fig, axs = plt.subplots(5, 5, figsize=(20, 15))
                    
                    fig.suptitle('Trajectories of the balls', y=0.95)
                    
                    plt.setp(axs[-1, :], xlabel='X-coordinate')
                    plt.setp(axs[:, 0], ylabel='Y-coordinate')

                    for row_id in range(len(axs)):
                        for col_id in range(len(axs[row_id])):
                            axs[row_id, col_id].set_xlim(0, 1)
                            axs[row_id, col_id].set_ylim(0, 1)

                    col_names = ['Blue ball', 'Green ball', 'Red ball', 'Scene (ground truth)', 'Scene (predicted)']

                    for ax, col in zip(axs[0], col_names):
                        ax.set_title(col)

                    for X, y in dataloader:
                        pred = model.run(X.squeeze().numpy())

                        for count in tqdm(range(5), desc='Plotting progress'):
                            gr = np.insert(y.squeeze().numpy()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][[0, 1, 3, 4, 6, 7]])
                            pr = np.insert(pred.squeeze()[count].reshape(1,-1)[0], 0, X.squeeze().numpy()[count][[0, 1, 3, 4, 6, 7]])

                            axs[count, 0].plot(gr[0::6], gr[1::6], color='blue', label='Ground truth')
                            axs[count, 0].plot(pr[0::6], pr[1::6], color='orange', label='Predicted')
                            axs[count, 0].legend()

                            axs[count, 1].plot(gr[2::6], gr[3::6], color='green', label='Ground truth')
                            axs[count, 1].plot(pr[2::6], pr[3::6], color='orange', label='Predicted')
                            axs[count, 1].legend()

                            axs[count, 2].plot(gr[4::6], gr[5::6], color='red', label='Ground truth')
                            axs[count, 2].plot(pr[4::6], pr[5::6], color='orange', label='Predicted')
                            axs[count, 2].legend()

                            data_linewidth_plot(gr[0::6], gr[1::6], ax=axs[count, 3], color='blue', linewidth=X.squeeze().numpy()[count][2], alpha=0.6)
                            data_linewidth_plot(gr[2::6], gr[3::6], ax=axs[count, 3], color='green', linewidth=X.squeeze().numpy()[count][5], alpha=0.6)
                            data_linewidth_plot(gr[4::6], gr[5::6], ax=axs[count, 3], color='red', linewidth=X.squeeze().numpy()[count][8], alpha=0.6)

                            data_linewidth_plot(pr[0::6], pr[1::6], ax=axs[count, 4], color='blue', linewidth=X.squeeze().numpy()[count][2], alpha=0.6)
                            data_linewidth_plot(pr[2::6], pr[3::6], ax=axs[count, 4], color='green', linewidth=X.squeeze().numpy()[count][5], alpha=0.6)
                            data_linewidth_plot(pr[4::6], pr[5::6], ax=axs[count, 4], color='red', linewidth=X.squeeze().numpy()[count][8], alpha=0.6)

                        break

                    plt.show()

                else:
                    print('Visual cannot be created as scenario cannot be identified from the data shape')
            else:
                print('Visual cannot be created as scenario cannot be identified from the data shape')
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
            parallels = parallels >> self.model[i]

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