import gym
import torch
from envs.deeprf.core import SLRExcitation, SLRInversion_origin, SLRInversion_OC
from scipy.io import loadmat, savemat
import numpy as np
import torch
import os

class DeepRFSLREXC20(gym.Env):
    def __init__(self, **kwargs):
        self.core = SLRExcitation(**kwargs)
        self.input_shape = self.core.input_shape
        self.action_space = self.core.action_space
        self.du = self.core.du
        self.max_amp = self.core.max_amp
        self.df = self.core.df
        self.sar_coef = kwargs.get('sar_coef', 0.0)
        self.ripple_coef = kwargs.get('ripple_coef', 1.0)
        self.max_mag = kwargs.get('max_mag', 0.9512)
        self.max_ripple = kwargs.get('max_ripple', 0.01)
        refer_Mxy  = loadmat('./refer_Mxy_512.mat')
        a = np.array(refer_Mxy['Mx'], dtype = np.float32)
        b = np.array(refer_Mxy['My'], dtype = np.float32)
        Mx = np.repeat(a, repeats = 101, axis= 0)
        My = np.repeat(b, repeats = 101, axis = 0 )
        
        self.refer_Mx = torch.from_numpy(Mx)
        self.refer_My = torch.from_numpy(My)


    def __len__(self):
        return self.core.sampling_rate

    def reset(self):
        return self.core.reset()

    def step(self, actions):
        m = actions[..., 0]
        phi = actions[..., 1]
        Mt, done = self.core.step(m, phi)
        amp = ((torch.clamp(m, -1.0, 1.0) + 1.0) * self.core.max_amp * 1e+4 / 2).pow(2) * \
              self.du / self.core.sampling_rate * 1e+6  # (mG)^2sec
        rewards = -self.sar_coef * amp
        if done:
            Mx = Mt[:,0,:,0]
            My = Mt[:,0,:,1]
            refer_Mx = self.refer_Mx.to(Mx.device)
            refer_My = self.refer_My.to(My.device)
            error = torch.mean((Mx - refer_Mx).square(),dim=1) + torch.mean((My-refer_My).square(),dim=1)
            rewards += -1*error
        return Mt.permute(0, 2, 1, 3), rewards, done

    
class DeepRFSLRINV_origin(gym.Env):
    def __init__(self, **kwargs):
        self.core = SLRInversion_origin(**kwargs)
        self.input_shape = self.core.input_shape
        self.action_space = self.core.action_space
        self.du = self.core.du
        self.max_amp = self.core.max_amp
        self.df = self.core.df
        self.sar_coef = kwargs.get('sar_coef', 0.0)
        self.ripple_coef = kwargs.get('ripple_coef', 1.0)
        self.max_mag = kwargs.get('max_mag', 0.9512)
        self.max_ripple = kwargs.get('max_ripple', 0.01)


    def __len__(self):
        return self.core.sampling_rate

    def reset(self):
        return self.core.reset()

    def step(self, actions):
        m = actions[..., 0]
        phi = actions[..., 1]
        Mt, done = self.core.step(m, phi)
        amp = ((torch.clamp(m, -1.0, 1.0) + 1.0) * self.core.max_amp * 1e+4 / 2).pow(2) * \
              self.du / self.core.sampling_rate * 1e+6  # (mG)^2sec
        rewards = -self.sar_coef * amp

        if done:
            Mz = Mt[:,0,:,2]
            refer_Mz = Mz[0,:]
            error = torch.mean((Mz - refer_Mz).square(),dim=1)
            rewards += -0.25*error
        return Mt.permute(0, 2, 1, 3), rewards, done
    
class DeepRFSLRINV_OC(gym.Env):
    def __init__(self, **kwargs):
        self.core = SLRInversion_OC(**kwargs)
        self.input_shape = self.core.input_shape
        self.action_space = self.core.action_space
        self.du = self.core.du
        self.max_amp = self.core.max_amp
        self.df = self.core.df
        self.sar_coef = kwargs.get('sar_coef', 0.0)
        self.ripple_coef = kwargs.get('ripple_coef', 1.0)
        self.max_mag = kwargs.get('max_mag', 0.9512)
        self.max_ripple = kwargs.get('max_ripple', 0.01)

    def __len__(self):
        return self.core.sampling_rate

    def reset(self):
        return self.core.reset()

    def step(self, actions):
        m = actions[..., 0]
        phi = actions[..., 1]
        Mt, done = self.core.step(m, phi)
        amp = ((torch.clamp(m, -1.0, 1.0) + 1.0) * self.core.max_amp * 1e+4 / 2).pow(2) * \
              self.du / self.core.sampling_rate * 1e+6  # (mG)^2sec
        rewards = -self.sar_coef * amp
        if done:
            Mx = Mt[:,0,:,0]
            My = Mt[:,0,:,1]
            Mz = Mt[:,0,:,2]
            refer_Mx = Mx[0,:]
            refer_My = My[0,:]
            refer_Mz = Mz[0,:]
            error = torch.mean((Mx - refer_Mx).square(),dim=1) + torch.mean((My-refer_My).square(),dim=1) + torch.mean((Mz-refer_Mz).square(),dim=1)
            rewards += -0.25/3*error
        return Mt.permute(0, 2, 1, 3), rewards, done