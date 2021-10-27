from os import path
import sys
sys.path.append(path.join(path.dirname(__file__), '..'))

import os
import time
import argparse
import gym
import numpy as np
from scipy.io import loadmat
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from Bloch import RF_simul, Bloch_simul_rot


def main(args):
    # Select GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Hyperparameter setting
    pulse_type = args.pulse_type
    gamma = 2 * pi * 42.5775 * 1e+6
    N_time_step = 256
    max_RF_amp = 0.2 * 1e-4 #T
    max_N_iter = args.episodes
    mu = args.mu       # learning rate
    alpha = args.alpha # trade off between slice profile loss and sar
    freq = np.linspace(-8000, 8000, 20000) #Hz
    T = args.du #sec
    k = freq.shape[0]
    max_rad = max_RF_amp * gamma * T./N_time_step
    time_step = T / N_time_step
    
    # create B matrix
    B1 = torch.zeros((3*k, 3*k)).to(device)
    B2 = torch.zeros((3*k, 3*k)).to(device)

    # create Projection matrix P
    P = torch.zeros(3*k, 3*k).to(device)

    for i in range(k) : 
        B1[i*3:(i+1)*3, i*3:(i+1)*3] = torch.from_numpy(np.array([[0,0,0], [0,0,1], [0,-1,0]]))
        B2[i*3:(i+1)*3, i*3:(i+1)*3] = torch.from_numpy(np.array([[0,0,-1], [0,0,0], [1,0,0]]))
        P[3*i, 3*i] = 1
        
    # Create input vectors
    preset = loadmat('./SLR_inv_oc.mat')
    slr = torch.unsqueeze(torch.from_numpy(np.array(preset['slr'], dtype=np.float32)), dim=0).to(device)
    rot = RF_simul(torch.from_numpy(slr), torch.zeros((256,1)).type(torch.DoubleTensor), torch.linspace(-8000,8000,2000), 5.12*1e-3/256).to(device)
    
    Mt = torch.zeros([k, N_time_step+1, 3]).to(device)
    Mt[:, :, 2] = torch.ones([k, N_time_step+1])
    
    for f in range(1, N_time_step+1):
        Mt[:, f, 0] = Mt[:, f-1, 0] * rot[:, f-1, 0, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 0, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 0, 2] 
        Mt[:, f, 1] = Mt[:, f-1, 0] * rot[:, f-1, 1, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 1, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 1, 2] 
        Mt[:, f, 2] = Mt[:, f-1, 0] * rot[:, f-1, 2, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 2, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 2, 2]
    
    D = torch.reshape(torch.transpose(Mt[:,-1,:].squeeze(), -1, 1)

    if args.preset is None:
        # Create random normal vector
        w1 = torch.FloatTensor(args.samples, args.sampling_rate).uniform_(-1.0, 1.0)
        w2 = torch.FloatTensor(args.samples, args.sampling_rate).uniform_(-1.0, 1.0)
    else:
        # Load preset and upsample if necessary
        preset = loadmat(args.preset)
        pulse = np.array(preset['result'], dtype=np.float32)
        w1 = torch.clamp(torch.FloatTensor(pulse[:args.samples, :, 0]), -1.0 + 1e-4, 1.0 - 1e-4)
        w2 = torch.FloatTensor(pulse[:args.samples, :, 1])

    w1 = w1.to(device)
    w2 = w2.to(device)
    
    # Optimal control
    st = time.time()
    for e in range(args.episodes):
        Mt = torch.zeros([k, N_time_step+1, 3])
        Mt[:, :, 2] = torch.ones([k, N_time_step+1])
        rot = (RF_simul(torch.transpose(w1,0,1), torch.transpose(w2),torch.transpose(freq), T/N_time_step)).squeeze()
        
        for f in range(1 ,N_time_step+1):
            Mt[:, f, 0] = Mt[:, f-1, 0] * rot[:, f-1, 0, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 0, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 0, 2] 
            Mt[:, f, 1] = Mt[:, f-1, 1] * rot[:, f-1, 1, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 1, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 1, 2] 
            Mt[:, f, 2] = Mt[:, f-1, 2] * rot[:, f-1, 2, 0] +  Mt[:, f-1, 1] * rot[:, f-1, 2, 1] +  Mt[:, f-1, 2] * rot[:, f-1, 2, 2] 
        
        Mt = Mt[:, 1:, :]
        
        K = torch.matmul(P, D);
        M_T = (Mt[:, end, :]).squeeze()
        M_T = torch.reshape(torch.transpose(M_T), (3*k, 1))
        lambda_T = torch.transpose(torch.matmul(torch.matmul(torch.transpose(P),P), M_T) - torch.matmul(torch.transpose(P), K))
        
        labmda = torch.zeros((k, N_time_step, 3))
        labmda[:, end, 0] = lambda_T[0:3:end]
        
        
        b1_ = torch.cat((ref_pulse, b1), dim=0)

        # Simulation
        t = 0
        done = False
        total_rewards = 0.0
        while not done:
            Mt, rews, done = env.step(b1_[..., t])
            t += 1
            total_rewards += rews
        env.reset()

        # Calculate loss from environment
        loss_ = -total_rewards[1:, ...]
        loss = loss_.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Update statistics
        diff = torch.mean(torch.square(Mt[1:, :args.pb, 0, 1] - Mt[0, :args.pb, 0, 1])+torch.square(Mt[1:, :args.pb, 0, 0] - Mt[0, :args.pb, 0, 0]), dim=1)
        difff = torch.mean(torch.square(Mt[1:, :, 0, 1] - Mt[0, :, 0, 1]) + torch.square(Mt[1:,:,0,0] - Mt[0,:,0,0]), dim=1)
        mxy_ref = torch.sqrt(Mt[0,:,0,1]**2 + Mt[0,:,0,0] **2).detach().cpu().numpy()
        mxy = torch.sqrt(Mt[:, :, 0, 1] **2 + Mt[:, :, 0, 0]**2).detach().cpu().numpy()
        pb = torch.sqrt(torch.sum(Mt[1:, :1000, 0, 0], dim=1) ** 2 + torch.sum(Mt[1:, :1000, 0, 1], dim=1) ** 2)
        ripple = torch.max(torch.sqrt(Mt[1:, 1000:, 0, 0] ** 2 + Mt[1:, 1000:, 0, 1] ** 2), dim=1)[0]
        amp = ((b1[:, 0, :] + 1.0) * env.max_amp * 1e4 / 2).pow(2).sum(-1)
        sar = amp * env.du / len(env) * 1e6
        

        idx1 = 0
        idx2 = 0
        idx3 = 0
        idx4 = 0
        idx5 = 0

        best_SAR1 = 100000
        best_SAR2 = 100000
        best_SAR3 = 100000
        best_SAR4 = 100000
        best_SAR5 = 100000

        for i in range(args.samples):
            if (diff[i]) < 1e-2 and sar[i] < best_SAR1 and ripple[i] < 0.01:
                idx1 = i
                best_SAR1 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-3 and sar[i] < best_SAR2 and ripple[i] < 0.01:
                idx2 = i
                best_SAR2 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-4 and sar[i] < best_SAR3 and ripple[i] < 0.01:
                idx3 = i
                best_SAR3 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-5 and sar[i] < best_SAR4 and ripple[i] < 0.01:
                idx4 = i
                best_SAR4 = sar[i]

        for i in range(args.samples):
            if (diff[i]) < 1e-6 and sar[i] < best_SAR5 and ripple[i] < 0.01:
                idx5 = i
                best_SAR5 = sar[i]

        # Log summary statistics
        if (e + 1) % args.log_step == 0:
            logger.log('Summary statistics for episode {}'.format(e + 1))

            # Excitation (magnitude) profile
            profile = plt.figure(1)
            plt.plot(np.concatenate((env.df[args.pb:args.pb + args.sb], env.df[:args.pb], env.df[args.pb + args.sb:args.pb + 2*args.sb])),
                     np.concatenate((mxy[idx3, args.pb:args.pb + args.sb], mxy[idx3, :args.pb], mxy[idx3, args.pb + args.sb:args.pb + 2*args.sb])))
            plt.plot(np.concatenate((env.df[args.pb:args.pb+ args.sb], env.df[:args.pb], env.df[args.pb + args.sb:args.pb + 2*args.sb])),
                     np.concatenate((mxy_ref[args.pb:args.pb+ args.sb], mxy_ref[:args.pb], mxy_ref[args.pb + args.sb:args.pb + 2*args.sb])), 'r')
            logger.image_summary(profile, e + 1, 'profile')

            # RF pulse magnitude
            fig_m = plt.figure(2)
            plt.plot(b1[idx3, 0, :].detach().cpu().numpy())
            plt.ylim(-1, 1)
            logger.image_summary(fig_m, e + 1, 'magnitude')

            # RF pulse phase
            fig_p = plt.figure(3)
            plt.plot(b1[idx3, 1, :].detach().cpu().numpy())
            logger.image_summary(fig_p, e + 1, 'phase')

            # matlab save
            array_dict = {'pulse': b1.detach().cpu().numpy(),
                          'sar': sar.detach().cpu().numpy(),
                          'loss_arr': loss_.detach().cpu().numpy(),
                          'ripple': ripple.detach().cpu().numpy(),
                          'difff': difff.detach().cpu().numpy(),
                          'diff': diff.detach().cpu().numpy(),
                          'pb': pb.detach().cpu().numpy()}
            logger.savemat('pulse' + str(e + 1), array_dict)

            logger.scalar_summary(info.val, e + 1)
            info.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="RF Pulse Design using Gradient Ascent"
    )
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=int(5e3))
    parser.add_argument("--sampling_rate", type=int, default=256)
    parser.add_argument("--preset", type=str, default=None)
    parser.add_argument("--pb", type=int, default=1400)
    parser.add_argument("--sb", type=int, default=1400)
    parser.add_argument("--du", type=float, default = 5.12e-3)
    parser.add_argument("--mu", type=float, dafault = 0.0001)
    args = parse_args(parser)
    main(args)
