"""
Code for reproducing the results of the paper: 'Compression Techniques For MIMO Channels in FDD Systems' (DSLW 2022)
@email: valentina.rizzello@tum.de

This script can be used to reproduce the results of Figure 5 of the paper.

Two arguments must be specified to run this file:
- type of VQ-layer: 'std' or 'conv'
- codebook size for VQ-layer: '2**3' or '2**6'


Examples:

    python3 main_train_ae_quant.py std 8

    python3 main_train_ae_quant.py std 64

    python3 main_train_ae_quant.py ema 8

    python3 main_train_ae_quant.py ema 64

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
from slvit_ae import *



def transform_to_tensor(x, device):
    x = torch.from_numpy(x)
    return x.to(device)



def train_epoch(model, optimizer, data_loader, reco_loss_history, commit_loss_history,
                perplexity_history, commit_cost=0.25, ema=False):
    total_samples = len(data_loader.dataset)
    model.train()

    running_reco_loss = 0.0
    running_commit_loss = 0.0
    running_perplexity = 0.0

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        if ema is not True:
            data_reco, quant_loss, commit_loss, perplexity = model(data)
        else:
            data_reco, commit_loss, perplexity = model(data)
            quant_loss = 0.0

        reco_loss = F.mse_loss(data_reco, data)

        loss = reco_loss + quant_loss + commit_cost * commit_loss
        loss.backward()

        optimizer.step()

        running_reco_loss += reco_loss.item()
        running_commit_loss += commit_loss.item()
        running_perplexity += perplexity.item()

        if i % 200 == 0:
            print \
                    (
                    '[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples)
                    + ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  train loss: '
                    + format(
                        loss.item(), '.3f'
                    )
                )

    reco_loss_history.append(running_reco_loss / len(data_loader))
    commit_loss_history.append(running_commit_loss / len(data_loader))
    perplexity_history.append(running_perplexity / len(data_loader))



def output(model, data_loader):
    model.eval()
    with torch.no_grad():
        i = 0
        for data, target in data_loader:
            if i == 0:
                pre = model(data)[0]
                true = target
            else:
                pre = torch.cat((pre, model(data)[0]), dim=0)
                true = torch.cat((true, target), dim=0)
            i += 1
        return true, pre



def evaluate(model, data_loader):
    true, pre = output(model, data_loader)

    nmse, mse, nmse_all = nmse_eval(true, pre)

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

    # load data
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

    foldername = 'vq_'

    for j, key in enumerate(config.keys()):
        foldername += f'{key}_{config[key]}_'


    if not os.path.exists(foldername):
        os.mkdir(foldername)

    model = SLViTAEQuant(
        dims_enc=[128, 64, 32],
        heads_enc=[config["h1"], config["h2"], config["h3"]],
        overall_dim=2048,
        p1=p1,
        p2=p2,
        hidden_dim=config["hidden_dim"],
        mode=config["mode"],
        num_embeddings=config["num_embeddings"],
        embedding_dim=1,
        decay=config["decay"]
    ).to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)

    # train the model
    reco_loss_history = []
    commit_loss_history = []
    perplexity_history = []

    es = EarlyStopping(patience=10, verbose=False, delta=0.001, path=os.path.join(foldername, 'checkpoint.pt'))

    n_epochs = 1000

    if config["decay"] > 0.0:
        ema = True
    else:
        ema = False

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        print('Epoch:', epoch)

        train_epoch(
            model, optimizer, train_loader, reco_loss_history, commit_loss_history, perplexity_history, config["beta"],
            ema
        )

        val_loss_nmse, val_loss_mse = evaluate(model, validation_loader)
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
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_ul'), nmse_ul.cpu())

    model.eval()
    # evaluate nmse for DL test set
    true_channels, pred_channels = output(model, dl_test_loader_1)
    nmse_test, mse_test, nmse_dl_1 = nmse_eval(true_channels, pred_channels)
    print('\n\n120 MHz frequency gap:', file=loss_file)
    print('\nafter training:', file=loss_file)
    print('\ntest loss', file=loss_file)
    print('nmse = ' + str(nmse_test.item()), file=loss_file)
    print('mse = ' + str(mse_test.item()), file=loss_file)

    np.save(os.path.join(foldername, 'nmse_dl_1'), nmse_dl_1.cpu())

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

    ax1.plot(reco_loss_history, label='reco loss')
    ax1.plot(commit_loss_history, label='commit loss')

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    title = 'model loss'
    ax1.set_title(title)
    ax1.legend(loc='best')

    plt.savefig(os.path.join(foldername, 'learning_curves.png'))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(perplexity_history, label="perplexity")
    title = 'perplexity'
    ax1.set_title(title)
    ax1.legend(loc='best')

    plt.savefig(os.path.join(foldername, 'perplexity.png'))



if __name__ == '__main__':
    """Run the script with the best configurations"""

    if sys.argv[1] == "ema":
        if sys.argv[2] == "8":
            config = {'patch_sizes_idx': 2, 'h1': 16, 'h2': 16, 'h3': 16, 'hidden_dim': 64, 'batch_size': 25,
                      'mode': 'linear', 'lr': 0.006015353710001343,
                      'beta': 0.2780677050121794, 'decay': 0.9505604755653548, 'num_embeddings': 8
                      }
        elif sys.argv[2] == "64":
            config = {'patch_sizes_idx': 2, 'h1': 16, 'h2': 16, 'h3': 16, 'hidden_dim': 64, 'batch_size': 25,
                      'mode': 'linear', 'lr': 0.0020005604375492943,
                      'beta': 0.19675492940423106, 'decay': 0.9671892237064758, 'num_embeddings': 64
                      }
    elif sys.argv[1] == "std":
        if sys.argv[2] == "8":
            config = {'patch_sizes_idx': 2, 'h1': 16, 'h2': 16, 'h3': 16, 'hidden_dim': 64, 'batch_size': 25,
                      'mode': 'linear', 'lr': 0.003165193035895348,
                      'beta': 0.21080855577190427, 'decay': 0.0, 'num_embeddings': 8
                      }
        elif sys.argv[2] == "64":
            config = {'patch_sizes_idx': 2, 'h1': 16, 'h2': 16, 'h3': 16, 'hidden_dim': 64, 'batch_size': 25,
                      'mode': 'linear', 'lr': 0.005288892606081014,
                      'beta': 0.15661906206256185, 'decay': 0.0, 'num_embeddings': 64
                      }

    main(config)