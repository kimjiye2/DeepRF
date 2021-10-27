import gym
import numpy as np
import torch
from envs.simulator import BlochSimulator
from settings import INF


class Bloch:
    def __init__(self):
        self.device = 'cuda' if torch.cuda_is_avaiable() else 'cpu'
        
    def RF_simul(RF_pulse, RF_img, off_range, time_step):
        size = RF_pulse.shape[0]
        RF_pulse_new = torch.zeros((size,2)).to(self.device)
        RF_pulse_new[:,0] = torch.sqrt((RF_pulse**2+RF_img**2).squeeze())
        RF_pulse_new[:,1] = torch.angle(torch.complex(RF_pulse, RF_img).squeeze())
        
        max_rf_amp = torch.max(RF_pulse_new[:,0]) / (2*np.pi *42.577*1e+6*time_step*1e-4)
        rr = 42.577; # MHz/T
        Gz =  40; # mT/m (fixed)
        pos = torch.abs(off_range[0] / rr / Gz); # mm
        mag = RF_pulse_new[:,0] / torch.max(RF_pulse_new[:,0]) * max_rf_amp
        ph = RF_pulse_new[:,1] / np.pi*180
        
        pulse = torch.cat([mag.unsqueeze(1), ph.unsqueeze(1)], 1)
        pulse = torch.transpose(pulse, 0, 1)
        gg = torch.transpose(torch.ones((RF_pulse_new.shape[0],1)), 0, 1) * Gz

        
        freq_shape = off_range.shape[0]
        rot = Bloch_simul_rot(np.zeros([freq_shape,1]),np.zeros([freq_shape,1]),np.ones([freq_shape,1]),1e+10,1e+10,pulse,gg,time_step * 1e+3,pos * 0.001,freq_shape,off_range)
        
        return rot
    
    def Bloch_simul_rot(x,y,z,T1,T2,RF,Gz,time_step,slice_thick,spatial_point,off_range):
        T1 = T1 / 1000

        T2 = T2 / 1000

        length_RF = RF[0,:].shape[0]
        t_int = time_step * 10 ** (- 3)

        # delta_omega = 2*pi*42.57747892*10^6 * (-1:2/(spatial_point-1):1).'*slice_thick*0.001*Gz;
        delta_omega = np.pi*off_range.unsqueeze(1).repeat(1,256)
        RF_amp = (RF[0,:] * 2 * np.pi * 4257.747892).repeat(spatial_point, 1)
        RF_phase = (RF[1,:]).repeat(spatial_point, 1)
        alpha = t_int * torch.sqrt(RF_amp ** 2 + delta_omega **2)
        zeta = torch.atan2(RF_amp, delta_omega)

        theta = RF_phase
        ca = torch.cos(alpha)
        print(ca.shape)
        sa = torch.sin(alpha)
        cz = torch.cos(zeta)
        sz = torch.sin(zeta)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        E1 = np.exp(- t_int / T1)
        E2 = np.exp(- t_int / T2)
        Mx_x_part = ct*(E2*ct*sz**2 + cz*(E2*sa*st + E2*ca*ct*cz)) + st*(E2*ca*st - E2*ct*cz*sa)
        Mx_y_part = st*(E2*ct*sz**2 + cz*(E2*sa*st + E2*ca*ct*cz)) - ct*(E2*ca*st - E2*ct*cz*sa)
        Mx_z_part = E2*ct*cz*sz - sz*(E2*sa*st + E2*ca*ct*cz)
        My_x_part = - ct*(- E2*st*sz**2 + cz*(E2*ct*sa - E2*ca*cz*st)) - st*(E2*ca*ct + E2*cz*sa*st)
        My_y_part = ct*(E2*ca*ct + E2*cz*sa*st) - st*(- E2*st*sz**2 + cz*(E2*ct*sa - E2*ca*cz*st))
        My_z_part = sz*(E2*ct*sa - E2*ca*cz*st) + E2*cz*st*sz
        Mz_x_part = ct*(E1*cz*sz - E1*ca*cz*sz) + E1*sa*st*sz
        Mz_y_part = st*(E1*cz*sz - E1*ca*cz*sz) - E1*ct*sa*sz
        Mz_z_part = E1*cz**2 + E1*ca*sz**2;

    
        rot = torch.zeros(spatial_point,length_RF,3,3).to(self.device)
        rot[:,:,0,0] = Mx_x_part
        rot[:,:,0,1] = Mx_y_part
        rot[:,:,0,2] = Mx_z_part
        rot[:,:,1,0] = My_x_part
        rot[:,:,1,1] = My_y_part
        rot[:,:,1,2] = My_z_part
        rot[:,:,2,0] = Mz_x_part
        rot[:,:,2,1] = Mz_y_part
        rot[:,:,2,2] = Mz_z_part
        return rot
