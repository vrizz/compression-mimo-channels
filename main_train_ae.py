"""
Code for reproducing the results of the paper: 'Compression Techniques For MIMO Channels in FDD Systems' (DSLW 2022)
@email: valentina.rizzello@tum.de

This script can be used to reproduce the results of Figure 3 of the paper.

Two arguments must be specified to run this file:
- type of architecture: 'slvit' or 'conv'
- mode of the architecture: 'linear' or 'nearest' for 'slvit'-type, 'strided', 'biliner' or 'pooling' for 'conv'-type


Example:

    python3 main_train_ae.py slvit linear

    python3 main_train_ae.py slvit nearest

    python3 main_train_ae.py conv strided

    python3 main_train_ae.py conv bilinear

    python3 main_train_ae.py conv pooling

After running the script the normalized-mean-squared-error for the downlink test set, which can be used to plot the CDF,
is saved in the file 'nmse_dl_1.npy' stored inside the folder of the simulated scenario.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from dataset_init_utils import get_dataset, get_dl_test_set
from pytorchtools import EarlyStopping



def transform_to_tensor(x, device):
    x = torch.from_numpy(x)
    return x.to(device)



def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)

    model.train()
    running_loss = 0.0

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target, reduction="sum")
        loss = loss / data.shape[0]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 0:
            print(
                '[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples)
                + ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  train loss (mse): '
                + format(loss.item(), '.3f')
            )

    loss_history.append(running_loss / len(data_loader))



def output(model, data_loader):
    model.eval()
    with torch.no_grad():
        i = 0
        for data, target in data_loader:
            if i == 0:
                pre = model(data)
                true = target
            else:
                pre = torch.cat((pre, model(data)), dim=0)
                true = torch.cat((true, target), dim=0)
            i += 1
        return true, pre



def evaluate(model, data_loader, loss_history):
    true, pre = output(model, data_loader)

    nmse, mse, nmse_all = nmse_eval(true, pre)

    loss_history.append(mse.item())

    print('\nvalidation loss (mse): ' + format(mse.item(), '.3f'))
    print('validation loss (nmse): ' + format(nmse.item(), '.3f'))
    print('----------------------------------------------------\n')
    return nmse, mse



def nmse_eval(y_true, y_pred):
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)

    mse = torch.sum(abs(y_pred - y_true) ** 2, dim=1)
    power = torch.sum(abs(y_true) ** 2, dim=1)

    nmse = mse / power
    return torch.mean(nmse), torch.mean(mse), nmse



def main(config, device_name="gpu", device_number="0"):

    if device_name == "cpu":
        device = "cpu"
    elif device_name == "gpu":
        device = "cuda:" + device_number

    x_train, x_val, x_test = get_dataset(direction='uplink', scenario_name='mimo_umi_nlos', gap='_')
    x_test_dl_1 = get_dl_test_set(scenario_name='mimo_umi_nlos', gap='1')

    # # dummy data
    # x_train = np.random.randn(1000, 2, 16, 64).astype("float32")
    # x_val = np.random.randn(400, 2, 16, 64).astype("float32")
    # x_test = np.random.randn(400, 2, 16, 64).astype("float32")
    # x_test_dl_1 = np.random.randn(400, 2, 16, 64).astype("float32")
    # # end dummy data

    x_train = transform_to_tensor(x_train, device)
    x_val = transform_to_tensor(x_val, device)
    x_test = transform_to_tensor(x_test, device)

    x_test_dl_1 = transform_to_tensor(x_test_dl_1, device)

    # create dataloader
    train_data_set = TensorDataset(x_train, x_train)
    val_data_set = TensorDataset(x_val, x_val)
    test_data_set = TensorDataset(x_test, x_test)

    dl_test_set_1 = TensorDataset(x_test_dl_1, x_test_dl_1)

    train_loader = DataLoader(train_data_set, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(val_data_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_data_set, batch_size=1000, shuffle=False)

    dl_test_loader_1 = DataLoader(dl_test_set_1, batch_size=1000, shuffle=False)

    if config['type'] == 'slvit':
        foldername = config['type'] + \
                     '_mode_' + config["mode"] + \
                     '_patch_size_idx_' + str(config["patch_sizes_idx"]) + \
                     '_h1_' + str(config["h1"]) + '_h2_' + str(config["h2"]) + '_h3_' + str(config["h3"]) + \
                     '_hidden_dim_' + str(config["hidden_dim"]) + '_lr_' + str(config["lr"]) + \
                     '_batch_size_' + str(config["batch_size"])

        patch_size_combinations = [
            [8, 8],
            [4, 16],
            [2, 32],
            [1, 64],
            [2, 64],
            [4, 32],
            [8, 16],
            [4, 8],
            [2, 16],
            [1, 32]
        ]

        idx = config["patch_sizes_idx"]

        p1 = patch_size_combinations[idx][0]
        p2 = patch_size_combinations[idx][1]

        model = SLViTAE(
            dims_enc=[128, 64, 32],
            heads_enc=[config["h1"], config["h2"], config["h3"]],
            overall_dim=2048,
            p1=p1,
            p2=p2,
            hidden_dim=config["hidden_dim"],
            mode=config["mode"]
        ).to(device=device)

    else:
        if config["mode"] == "strided":
            foldername = config['type'] + \
                         '_mode_' + config["mode"] + \
                         '_filter1_' + str(config["filter1"]) + \
                         '_filter2_' + str(config["filter2"]) + '_filter3_' + str(config["filter3"]) + \
                         'k1_' + str(config["k1"]) + \
                         '_k2_' + str(config["k2"]) + '_k3_' + str(config["k3"]) + \
                         '_strides_idx_' + str(config["strides_idx"]) + \
                         '_lr_' + str(config["lr"]) + \
                         '_batch_size_' + str(config["batch_size"])

            strides_combinations = [
                [2, 4, 2],
                [4, 2, 2],
                [2, 2, 2],
                [4, 2, 1],
                [2, 2, 1]
            ]

            enc_strides = strides_combinations[config["strides_idx"]]
            model = ConvNet(
                latent_dim=512, filter1=config["filter1"], filter2=config["filter2"], filter3=config["filter3"],
                enc_kernel_ls=[config["k1"], config["k2"], config["k3"]],
                enc_strides=enc_strides, overall_dim=1024
            ).to(device=device)

        else:
            foldername = config['type'] + \
                         '_mode_' + config["mode"] + \
                         '_filter1_' + str(config["filter1"]) + \
                         '_filter2_' + str(config["filter2"]) + '_filter3_' + str(config["filter3"]) + \
                         'k1_' + str(config["k1"]) + \
                         '_k2_' + str(config["k2"]) + '_k3_' + str(config["k3"]) + \
                         '_lr_' + str(config["lr"]) + \
                         '_batch_size_' + str(config["batch_size"])

            if config["mode"] == "pooling":
                model = ConvNet(
                    latent_dim=512, filter1=config["filter1"], filter2=config["filter2"], filter3=config["filter3"],
                    enc_kernel_ls=[config["k1"], config["k2"], config["k3"]],
                    enc_scaling_factors=[0.5, 0.5, 0.5], pooling=True
                ).to(device=device)

            elif config["mode"] == "bilinear":
                model = ConvNet(
                    latent_dim=512, filter1=config["filter1"], filter2=config["filter2"], filter3=config["filter3"],
                    enc_kernel_ls=[config["k1"], config["k2"], config["k3"]],
                    enc_scaling_factors=[0.5, 0.5, 0.5], pooling=False
                ).to(device=device)

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    # train the model
    train_loss_history = []
    val_loss_history = []
    es = EarlyStopping(patience=10, verbose=False, delta=0.001, path=os.path.join(foldername, 'checkpoint.pt'))

    n_epochs = 1000

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)

        train_epoch(model, optimizer, train_loader, train_loss_history)
        val_loss_nmse, val_loss_mse = evaluate(model, validation_loader, val_loss_history)
        scheduler.step(val_loss_mse)

        # early stopping
        es(val_loss_mse, model)
        if es.early_stop:
            print('early stopping')
            break

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
    print('----------------------------------------------------')

    model.load_state_dict(torch.load(os.path.join(foldername, 'checkpoint.pt')))

    loss_file = open(os.path.join(foldername, 'loss_data.txt'), 'w')

    # evaluate NMSE
    model.eval()
    # test loss
    true_channels, pred_channels = output(model, test_loader)
    nmse_test, mse_test, nmse_ul = nmse_eval(true_channels, pred_channels)

    print('\nafter training:', file=loss_file)
    print('\n\n UL loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_ul'), nmse_ul.cpu())

    model.eval()
    # evaluate nmse for DL test set
    true_channels, pred_channels = output(model, dl_test_loader_1)
    nmse_test, mse_test, nmse_dl_1 = nmse_eval(true_channels, pred_channels)
    print('\n\nDL (120 MHz frequency gap) loss:', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_dl_1'), nmse_dl_1.cpu())

    model.eval()

    # plot an example
    index = 10
    true_ex = true_channels[index, 0].detach().cpu().numpy()
    pred_ex = pred_channels[index, 0].detach().cpu().numpy()

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.imshow(true_ex)
    ax1.set_title('true channel')
    ax2.imshow(pred_ex)
    ax2.set_title('predicted channel')

    plt.tight_layout()
    plt.savefig(os.path.join(foldername, 'Real part channel before and after training'))

    # learning curves
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    epochs = np.linspace(1, len(train_loss_history), len(train_loss_history)).astype('int')

    ax1.plot(epochs, train_loss_history, label='train loss')
    ax1.plot(epochs, val_loss_history, label='validation loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    title = 'model loss'
    ax1.set_title(title)
    ax1.legend(loc='best')

    plt.savefig(os.path.join(foldername, 'learning curves.png'))



if __name__ == '__main__':
    """Run the script with the best configurations"""

    if sys.argv[1] == 'slvit':
        from slvit_ae import *


        if sys.argv[2] == 'linear':
            config = {
                'patch_sizes_idx': 2, 'h1': 16, 'h2': 16, 'h3': 16, 'hidden_dim': 64, 'lr': 0.0017427275993200173,
                'batch_size': 25, 'mode': 'linear', 'type': sys.argv[1]
            }
        elif sys.argv[2] == 'nearest':
            config = {
                'patch_sizes_idx': 2, 'h1': 16, 'h2': 32, 'h3': 4, 'hidden_dim': 128, 'lr': 0.0028992186596203096,
                'batch_size': 25, 'mode': 'nearest', 'type': sys.argv[1]
            }
    elif sys.argv[1] == 'conv':
        if sys.argv[2] == 'strided':
            from conv_strides_ae import *


            config = {
                'filter1': 8, 'filter2': 64, 'filter3': 64, 'k1': 5, 'k2': 3, 'k3': 3, 'strides_idx': 4,
                'lr': 0.0010984585616922759, 'batch_size': 50, 'mode': 'strided', 'type': sys.argv[1]
            }
        elif sys.argv[2] == "pooling":
            from conv_ae import *


            config = {
                'filter1': 16, 'filter2': 64, 'filter3': 128, 'k1': 7, 'k2': 5, 'k3': 3,
                'lr': 0.0008284483622897589, 'batch_size': 25, 'mode': 'pooling', 'type': sys.argv[1]
            }

        elif sys.argv[2] == "bilinear":
            from conv_ae import *


            config = {
                'filter1': 32, 'filter2': 64, 'filter3': 128, 'k1': 5, 'k2': 3, 'k3': 3,
                'lr': 0.00098327127823576, 'batch_size': 25, 'mode': 'bilinear', 'type': sys.argv[1]
            }

    main(config)