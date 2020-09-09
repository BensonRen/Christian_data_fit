"""
Wrapper functions for the networks
"""
# Built-in
import os
import time

# Torch
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
#from torchsummary import summary
from torch.optim import lr_scheduler
#from torchviz import make_dot
#from network_model import Lorentz_layer
from plotting_functions import plot_weights_3D, plotMSELossDistrib, \
    compare_spectra, compare_Lor_params

# Libs
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelmax



class Network(object):
    def __init__(self, model_fn, flags, train_loader, test_loader,
                 ckpt_dir=os.path.join(os.path.abspath(''), 'models'),
                 inference_mode=False, saved_model=None):
        self.model_fn = model_fn                                # The network architecture object
        self.flags = flags                                      # The flags containing the hyperparameters
        if inference_mode:                                      # If inference mode, use saved model
            self.ckpt_dir = os.path.join(ckpt_dir, saved_model)
            self.saved_model = saved_model
            print("This is inference mode, the ckpt is", self.ckpt_dir)
        else:                                                   # Network training mode, create a new ckpt folder
            if flags.model_name is None:                    # Use custom name if possible, otherwise timestamp
                self.ckpt_dir = os.path.join(ckpt_dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
            else:
                self.ckpt_dir = os.path.join(ckpt_dir, flags.model_name)
        self.model = self.create_model()                        # The model itself
        self.loss = self.make_custom_loss()                            # The loss function
        self.optm = None                                        # The optimizer: Initialized at train()
        self.lr_scheduler = None                                # The lr scheduler: Initialized at train()
        self.train_loader = train_loader                        # The train data loader
        self.test_loader = test_loader                          # The test data loader
        self.log = SummaryWriter(self.ckpt_dir)     # Create a summary writer for tensorboard
        self.best_validation_loss = 0.1    # Set the BVL to large number
        self.best_lor_param_dict = None    # The best lorentz parameter for current spectra
        self.best_pretrain_loss = float('inf')
        self.running_loss = []
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = "cpu"
        # self.pre_train_model = self.flags.pre_train_model

    def create_model(self):
        """
        Function to create the network module from provided model fn and flags
        :return: the created nn module
        """
        model = self.model_fn(self.flags)
        # summary(model, input_data=(8,))
        print(model)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('There are %d trainable out of %d total parameters' %(pytorch_total_params, pytorch_total_params_train))
        return model

    def make_MSE_loss(self, logit=None, labels=None):
        """
        Create a tensor that represents the loss. This is consistent both at training time \
        and inference time for a backward model
        :param logit: The output of the network
        :return: the total loss
        """
        if logit is None:
            return None
        MSE_loss = nn.functional.mse_loss(logit, labels)          # The MSE Loss of the network
        return MSE_loss

    def mirror_padding(self, input, pad_width=1):
        """
        pads the input tensor by mirroring
        :param input: The tensor to be padded
        :param pad_width: The padding width (default to be 1)
        :return: the padded tensor
        """
        # Get the shape and create new tensor
        shape = np.array(np.shape(input.detach().cpu().numpy()))
        shape[-1] += 2*pad_width
        padded_tensor = torch.zeros(size=tuple(shape))
        #print("shape in mirror: ", np.shape(padded_tensor))
        padded_tensor[:, pad_width:-pad_width] = input
        padded_tensor[:, 0:pad_width] = input[:, pad_width:2*pad_width]
        padded_tensor[:, -pad_width:] = input[:, -2*pad_width:-pad_width]
        if torch.cuda.is_available():
            padded_tensor = padded_tensor.cuda()
        return padded_tensor

    # Peak finder loss
    def peak_finder_loss(self, logit=None, labels=None, w0=None, w_base=None):
        batch_size = labels.size()[0]
        #batch_size = 1
        loss_penalty = 100
        # Define the convolution window for peak finding
        descend = torch.tensor([0, 1, -1], requires_grad=False, dtype=torch.float32)
        ascend = torch.tensor([-1, 1, 0], requires_grad=False, dtype=torch.float32)
        # Use GPU option
        if torch.cuda.is_available():
            ascend = ascend.cuda()
            descend = descend.cuda()
        # make reflection padding
        padded_labels = self.mirror_padding(labels)
        # Get the maximum and minimum values
        max_values = F.conv1d(padded_labels.view(batch_size, 1, -1),
                             ascend.view(1, 1, -1), bias=None, stride=1)
        min_values = F.conv1d(padded_labels.view(batch_size, 1, -1),
                                     descend.view(1, 1, -1), bias=None, stride=1)
        min_values = F.relu(min_values)
        #max_values[max_values == 0] = 1
        max_values = F.relu(max_values)

        # Get the peaks
        zeros = torch.mul(max_values, min_values).squeeze() > 0
        ###############################
        ###############################
        peaks = torch.zeros(size=[batch_size, 4], requires_grad=False, dtype=torch.float32)
        for i in range(batch_size):
            peak_current = w_base[zeros[i, :]]
            #print("len of peak_current: ", len(peak_current))
            peak_num = len(peak_current)
            if peak_num == 4:
                peaks[i, :] = peak_current
            else:
                peak_rank, index = torch.sort(labels[i, zeros[i, :]])  # Get the rank of the peaks
                peaks[i, :peak_num] = peak_current                  # put the peaks into first len spots
                peaks[i, peak_num:] = peak_current[index[0]]        # make the full array using the highest peak
        #peaks = torch.tensor(w_base_expand[zeros], requires_grad=False, dtype=torch.float32)
        if torch.cuda.is_available():
            peaks = peaks.cuda()
        # sort the w0 to match the peak orders
        w0_sort, indices = torch.sort(w0)
        #print("shape of w0_sort ", np.shape(w0_sort))
        #print("shape of peaks ", np.shape(peaks))
        #print(w0_sort)
        #print(peaks)
        return loss_penalty * F.mse_loss(w0_sort, peaks)

    def make_custom_loss(self, logit=None, labels=None, w0=None,
                         g=None, wp=None, epoch=None, gradient_descend=True):
        """
        The custom master loss function
        :param logit: The model output
        :param labels: The true label
        :param w0: The Lorentzian parameter output w0
        :param g: The Lorentzian parameter output g
        :param wp: The Lorentzian parameter output wp
        :param epoch: The current epoch number
        :param peak_loss: Whether use peak_finding_loss or not
        :param gt_lor: The ground truth Lorentzian parameter
        :param lor_ratio: The ratio of lorentzian parameter to spectra during training
        :param lor_loss_only: The flag to have only lorentzian loss, this is for alternative training
        :param gt_match_style: The style of matching the GT Lor param to the network output:
              'gt': The ground truth correspondence matching
              'random': The permutated matching in random
              'peak': Match according to the sorted peaks (w0 values sequence)
        :param gradient_descend: The flag of gradient descend or ascend
        :return:
        """
        if logit is None:
            return None

        ############
        # MSE Loss #
        ############
        custom_loss = nn.functional.mse_loss(logit, labels, reduction='mean')

        # Gradient ascent
        if gradient_descend is False:
            custom_loss *= -self.flags.gradient_ascend_strength

        ######################
        # Boundary loss part #
        ######################
        if w0 is not None:
            freq_mean = (self.flags.freq_low + self.flags.freq_high)/ 2
            freq_range = (self.flags.freq_high - self.flags.freq_low)/ 2
            custom_loss += torch.sum(torch.relu(torch.abs(w0 - freq_mean) - freq_range))
        if g is not None:
            if epoch is not None and epoch < 100:
                custom_loss += torch.sum(torch.relu(-g + 0.05))
            else:
                custom_loss += 100 * torch.sum(torch.relu(-g))
        if wp is not None:
            custom_loss += 100*torch.sum(torch.relu(-wp))
        return custom_loss


    def make_optimizer(self, param=None):
        """
        Make the corresponding optimizer from the flags. Only below optimizers are allowed.
        :return:
        """
        if param is None:
            param = self.model.parameters()
        if self.flags.optim == 'Adam':
            op = torch.optim.Adam(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'AdamW':
            op = torch.optim.AdamW(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'Adamax':
            op = torch.optim.Adamax(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SparseAdam':
            op = torch.optim.SparseAdam(param, lr=self.flags.lr)
        elif self.flags.optim == 'RMSprop':
            op = torch.optim.RMSprop(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale)
        elif self.flags.optim == 'SGD':
            op = torch.optim.SGD(param, lr=self.flags.lr, weight_decay=self.flags.reg_scale, momentum=0.9, nesterov=True)
        elif self.flags.optim == 'LBFGS':
            op = torch.optim.LBFGS(param, lr=1, max_iter=20, history_size=100)
        else:
            raise Exception("Optimizer is not available at the moment.")
        return op

    def make_lr_scheduler(self):
        """
        Make the learning rate scheduler as instructed. More modes can be added to this, current supported ones:
        1. ReduceLROnPlateau (decrease lr when validation error stops improving
        :return:
        """
        # return lr_scheduler.StepLR(optimizer=self.optm, step_size=50, gamma=0.75, last_epoch=-1)
        try:
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm_all, mode='min',
                                        factor=self.flags.lr_decay_rate,
                                          patience=10, verbose=True, threshold=1e-4)
        except:
            return lr_scheduler.ReduceLROnPlateau(optimizer=self.optm, mode='min',
                                                  factor=self.flags.lr_decay_rate,
                                                  patience=10, verbose=True, threshold=1e-4)

    def reset_lr(self, optm):
        """
        Reset the learning rate to to original lr
        :param optm: The optimizer
        :return: None
        """
        self.lr_scheduler = self.make_lr_scheduler()
        for g in optm.param_groups:
            g['lr'] = self.flags.lr

    def train_stuck_by_lr(self, optm, lr_limit):
        """
        Detect whether the training is stuck with the help of LR scheduler which decay when plautue
        :param optm: The optimizer
        :param lr_limit: The limit it judge it is stuck
        :return: Boolean value of whether it is stuck
        """
        for g in optm.param_groups:
            if g['lr'] < lr_limit:
                return True
            else:
                return False


    def initialize_lor_params(self):
        """
        Initialize the lorentzian parameters
        :return: The lorentzian parameters
        """
        w0 = torch.tensor(np.random.uniform(self.flags.freq_low, self.flags.freq_high, self.flags.num_lor), requires_grad=True,
                          device=self.device)
        wp = torch.tensor(np.random.uniform(0, 5, self.flags.num_lor), requires_grad=True, device=self.device)
        g = torch.tensor(np.random.uniform(0, 0.05, self.flags.num_lor), requires_grad=True, device=self.device)
        eps_inf = torch.tensor(10., requires_grad=True, device=self.device)
        d = torch.tensor(50., requires_grad=True, device=self.device)
        return w0, wp, g, eps_inf, d

    def save_lor_params(self, w0, wp, g, eps_inf, d):
        """
        To save the lor params into a dictionary for future plotting
        Inputs: Pytorch tensors on GPU
        :return: A dictionary that keeps the lor params, on CPU with deep copy
        """
        lor_param_dict = {}
        lor_param_dict['w0'] = np.copy(w0.detach().cpu().numpy())
        lor_param_dict['wp'] = np.copy(wp.detach().cpu().numpy())
        lor_param_dict['g'] = np.copy(g.detach().cpu().numpy())
        lor_param_dict['eps_inf'] = np.copy(eps_inf.detach().cpu().numpy())
        lor_param_dict['d'] = np.copy(d.detach().cpu().numpy())
        self.best_lor_param_dict = lor_param_dict
        return lor_param_dict

    def train(self):
        """
        The major training function. This starts the training using parameters given in the flags
        :return: None
        """
        print("Starting training process")
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.model.cuda()
        # for epoch in range(self.flags.train_step):         # Normal training
        for j, (geometry, spectra) in enumerate(self.train_loader):
            # Initialized the parameters
            w0, wp, g, eps_inf, d = self.initialize_lor_params()
            self.best_lor_param_dict = None
            self.best_loss = self.best_validation_loss
            # Set the training flag to be True at the start, this is for the gradient ascend
            train_flag = True
            epoch = 0
            gradient_descend = True

            if cuda:
                spectra = spectra.double().squeeze().cuda()  # Put data onto GPU
                w0 = w0.cuda()
                wp = wp.cuda()
                g = g.cuda()
                eps_inf = eps_inf.cuda()
                d = d.cuda()

            # Set up the optimizer
            all_param = [w0, wp, g, eps_inf, d]  # Group them up for optimizer

            # Construct optimizer after the model moved to GPU
            self.optm_all = self.make_optimizer(param=all_param)
            self.lr_scheduler = self.make_lr_scheduler()

            while train_flag:
                epoch += 1

                if gradient_descend is False:
                    print('This is Epoch {} doing gradient ascend to avoid local minimum'.format(epoch))
                # Set to Training Mode
                train_loss = []
                train_loss_eval_mode_list = []
                self.model.train()

                self.optm_all.zero_grad()                                   # Zero the gradient first
                #print("mean of spectra target", np.mean(spectra.data.numpy()))
                logit, e2, e1 = self.model(w0, wp, g, eps_inf, d)            # Get the output

                loss = self.make_custom_loss(logit=logit, labels=spectra,
                                        w0=w0, wp=wp, g=g)
                # print(loss)
                loss.backward()

                self.optm_all.step()  # Move one step the optimizer

                train_loss.append(np.copy(loss.cpu().data.numpy()))     # Aggregate the loss
                self.running_loss.append(np.copy(loss.cpu().data.numpy()))

                # Calculate the avg loss of training
                train_avg_loss = np.mean(train_loss)
                if not gradient_descend:
                    train_avg_loss *= -1
                #train_avg_eval_mode_loss = np.mean(train_loss_eval_mode_list)

                # Validation part
                if epoch % self.flags.eval_step == 0 or not gradient_descend:           # For eval steps, do the evaluations and tensor board
                    print('Epoch {}, loss is {}'.format(epoch, loss.detach().cpu().numpy()))

                if gradient_descend:                # If currently in gradient descend mode
                    # # Learning rate decay upon plateau
                    self.lr_scheduler.step(train_avg_loss)
                    loss_numpy = loss.detach().cpu().numpy()
                    # Save the lor params if that is the current best
                    if loss_numpy < self.best_loss:
                        self.save_lor_params(w0, wp, g, eps_inf, d)
                    if loss_numpy > 0.001:
                        # If the LR changed (i.e. training stuck) and also loss is large
                        if self.train_stuck_by_lr(self.optm_all, self.flags.lr/8):
                            # Switch to the gradient ascend mode
                            gradient_descend = False
                    else:
                        print("The loss is lower than 0.01! good news")
                        # Stop the training
                        train_flag = False
                        self.save_lor_params(w0, wp, g, eps_inf, d)
                        #print("Saving the model...")
                else:                               # Currently in ascent mode, change to gradient descend mode
                    print("After the gradient ascend, switching back to gradient descend")
                    gradient_descend = True         # Change to Gradient descend mode
                    self.reset_lr(self.optm_all)    # reset lr
                if epoch > self.flags.train_step:
                    train_flag = False

            # After the while loop, we got our best fit for the current spectra now
            self.plot_spectra_compare(spectra, j)

        self.log.close()

    def plot_spectra_compare(self, spectra, j, save_dir='data'):
        """
        Plot the comparison spectra
        :param best_lor_param_dict: The dictionary that contains the best_lor_parameters
        :param model:  The Lorentz model
        :param spectra: The gt spectra
        :param j: The spectra number to save
        :return: Plotting of the best case
        """
        # Get the lor params out
        w0 = torch.tensor(self.best_lor_param_dict['w0'], device=self.device)
        wp = torch.tensor(self.best_lor_param_dict['wp'], device=self.device)
        g = torch.tensor(self.best_lor_param_dict['g'], device=self.device)
        d = torch.tensor(self.best_lor_param_dict['d'], device=self.device)
        eps_inf = torch.tensor(self.best_lor_param_dict['eps_inf'], device=self.device)

        # Get spectra from model
        logit, e2, e1 = self.model(w0, wp, g, eps_inf, d)

        # Plot the comparison
        f = plt.figure()
        w = self.model.w.cpu().numpy()
        plt.plot(w, logit.detach().cpu().numpy(), label='pred')
        plt.plot(w, spectra.cpu().numpy(), label='gt')
        plt.legend()
        plt.ylim([0, 1])
        plt.ylabel('T')
        plt.xlabel('fre')
        plt.savefig(os.path.join(save_dir, 'lor_num_{}_fit_spectra_plot_{}.png'.format(self.flags.num_lor, j)))