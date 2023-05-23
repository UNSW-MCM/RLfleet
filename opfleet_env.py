# Module containing the fleet environment and all its associated methods/functions
"""
Created on Wed Oct  6 15:27:42 2021

@author: Kilian Vos
"""

import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib import gridspec
import seaborn as sns
import pickle
import pdb
import gym
from gym import spaces
import random
import time, copy, itertools
from collections import deque
import shutil
# Pytorch
import torch
from torch.autograd import Variable
import time

#%% 1. Degradation model

def degradation_model(ds,nt,a_max,seed=0,C=[],levels=0):
    
    # inputs to degradation model
    # ds = stress range in MPa, depends on manoeuvre
    # nt = number of trajectories
    # a_max = maximum crack length allowable in mm
    # seed = random seed to draw the C values
    # C = values of C already provided (by default draw randomly)
    
    # constants for Paris Law (identified from Virkler dataset)
    b = 76.2/1000               # body width converted to metres
    m = 2.9                     # Paris exponent
    C_median = 8.586e-11        # median of Paris coefficient C
    C_std = 0.619e-11           # std of Paris coefficient C
    a_initial = 9               # initial crack length in mm
    a_step = 0.1                # step in crack length in mm
    
    use_C = False
    if len(C) == 0:
        # generate random C values from distribution (one for each trajectory)
        np.random.seed(seed)
        C = C_median + C_std*np.random.normal(size=nt)
    else:
        use_C = True
        nt = len(C)
        print('using provided C values')
    Cmax = np.max(C)
        
    # plot distribution of C
    # plt.figure()
    # plt.hist(C,bins=50)
    # plt.axvline(x=C_median,c='r')
    # plt.axvline(x=C_median+C_std,c='g')
    # plt.axvline(x=C_median-C_std,c='g')
    # plt.gcf().savefig('distribution_C.jpg')
    
    # generate degradation data
    a, step = np.linspace(a_initial,a_max,
                          np.round((a_max-a_initial)/a_step).astype(int)+1,
                          endpoint=True, retstep=True)
    # convert crack length to metres
    a = a /1000
    
    # compute trajectories
    dN = np.empty([len(a)-1,len(C)])
    dN_min = np.empty(len(a)-1)
    for i in range(len(a)-1):
        num = a[i+1]-a[i]
        for k in range(len(C)):
            denom = (C[k]*(ds*np.sqrt(np.pi*a[i]/np.cos(np.pi*a[i]/(2*b))))**m)
            dN[i,k] = num/denom
        denom_max = (Cmax*(ds*np.sqrt(np.pi*a[i]/np.cos(np.pi*a[i]/(2*b))))**m)
        dN_min[i] = num/denom_max
    N = np.concatenate([np.zeros([1,len(C)]),np.cumsum(dN,axis=0)],axis=0)     
    N_min = np.append(0,np.cumsum(dN_min))
    
    title = 'nt = %d, ds = %.2f Mpa, a_max = %.1f mm'%(nt,ds,a_max)
    label = 'Nmin    = %d,000 cycles\nNmax   = %d,000 cycles\nNmean = %d,000 cycles'%(np.min(N[-1,:])/1000,
                                                                                      np.max(N[-1,:])/1000,
                                                                                      np.mean(N[-1,:])/1000)
    fig,ax = plt.subplots(2,1,figsize=[9.37, 7.78],tight_layout=True,sharex=True,
                          gridspec_kw={'height_ratios':[1,4],'hspace':0})
    ax[0].grid(which='both',ls=':',c='0.5',lw=1)
    ax[1].grid(which='both',ls=':',c='0.5',lw=1)
    ax[0].set(title='Degradation model ( %s )'%title)
    ax[0].yaxis.set_label_position("right")
    ax[0].yaxis.tick_right()
    ax[1].set(ylabel='crack length [mm]',xlabel='number of cycles N')
    sns.histplot(N[-1,:],element='step',fc='0.75',ec='k',ax=ax[0],
                 label=label)
    ax[0].legend(loc='center left',frameon=True)
    if not use_C: 
        ax[1].axhline(y=a_initial,lw=2,ls='--',c='g',label='initial crack length = %.1f mm'%a_initial)
        ax[1].axhline(y=a_max,lw=2,ls='--',c='r',label='critical crack length = %.1f mm'%a_max)
        for i in range(1,len(C)):
            ax[1].plot(N[:,i],a*1000,c='0.5',alpha=0.5)
        ax[1].plot(N_min,a*1000,c='C0',ls='--',lw=2,alpha=0.75,label='max(C)')
        idx_min = np.argmin(C)
        ax[1].plot(N[:,idx_min],a*1000,c='C1',ls='--',lw=2,alpha=0.75,label='min(C)')
    else:
        ax[1].axhline(y=a_initial,lw=2,ls='--',c='k')
        ax[1].axhline(y=a_max,lw=2,ls='-',c='k')
        colors = sns.color_palette('Dark2',len(C)) 
        for i in range(len(C)):
            ax[1].plot(N[:,i],a*1000,c=colors[i],ls='-',lw=2,
                       label='tail #%d'%(i+1))
    ax[1].legend(loc='center left',frameon=True)
    
    if levels > 0:
        cmap = plt.get_cmap('YlOrRd',levels)
        colors = [cmap(_) for _ in np.linspace(0,1,levels)]
        intervals = np.linspace(a_initial,a_max,levels)
        for k in range(1,len(intervals)):
            ax[1].axhspan(ymin=intervals[k-1],ymax=intervals[k],fc=colors[k],
                            alpha=0.25)
            if k < len(intervals)-1:
                ax[1].axhline(y=intervals[k],ls='-',c='k',lw=0.5,alpha=0.75)
    
    return N, N_min, a, C, fig

# Paris law to propagate the crack length
def propagate_crack(a0,period,ds,C,f0,b=76.2/1000,m=2.9):
    a_increment = f0*period*(C*(ds*np.sqrt(np.pi*a0/np.cos(np.pi*a0/(2*b))))**m)
    a_new = a0 + a_increment
    return a_new

#%% 2. Environment Class

# Env class for RL environment with states, actions, rewards and transitions
class Env(gym.Env):
    """
    Description:
        An environment to simulate the operation of a fleet of aircrafts.
        Each day, a set of missions are prescribed to a fleet of aircrafts.
        The agent can choose for each aircraft to assign a mission, standby or maintenance.
        Maintenance is automatically applied when the aircraft reaches the critical crack length.
        The goal is to maximise the availability of the fleet, which is the rate of mission completion over 20 years.

    Observation space:
        - damage status of each tail number [9mm to 38.1mm]
        - maintenance status of each tail number [number of days of maintenance remaining, 0 if available]
        - prescribed missions for the day [number of missions of each type to complete at each timestep]
    Actions:
        - fly one of n missions [M1, M2, M_n]
        - standby
        - preventive maintenance
    Rewards:
        The reward is +1 for M1, +2 for M2, +n for M_n 
        -10 for preventive maintenance
        -100 for corrective maintenance
                        
    Starting State:
        Multiple groups of aircrafts with different service history
    Episode Termination:
        after 5 years of operation (2018 to 2023)
        
    """

    def __init__(self,params,verbose=False):
        "initialise class"
        # degradation model parameters
        self.b = params['b']
        self.m = params['m']
        self.C_median = params['C_median']
        self.C_std = params['C_std']
        self.a0 = params['a0']
        self.amax = params['amax']
        self.f0 = params['f0']

        # discretise crack length
        self.dlevels = params['damage_levels']
        self.dintervals =  np.linspace(self.a0*1000,self.amax*1000,self.dlevels)
                
        # mission parameters
        self.maneuvers = params['maneuvers']
        self.missions = params['missions']
        # calculate stress range per mission (f0*ds*dt)
        self.stress_per_mission = []
        for key in self.missions.keys():
            damage = 0
            for k,ds in enumerate(params['missions'][key]):
                damage += self.f0*ds*self.maneuvers[k]
            self.stress_per_mission.append(damage)
            
        # fleet composition
        self.n_tail = params['n_tail']
        
        if 'groups_tail' in params.keys():
            self.group_tail = params['groups_tail']
        else:
            # each tail number is a squadron
            self.group_tail = dict({})
            count = 0
            for k in range(self.n_tail):
                self.group_tail['group%d'%(k+1)] = np.arange(count,k+1)
                count = k+1
                
        # maintenance waiting period and repair level
        # self.maintenance_levels = params['maintenance_levels']
        self.corrective_maintenance = params['corrective_maintenance']
        self.preventive_maintenance = params['preventive_maintenance']
        
        # generate randomly one C coefficient for each aircraft
        # np.random.seed(self.seed)
        # self.C = self.C_median + self.C_std*np.random.normal(size=self.n_tail)
        # or load C values directly (to always maintain the same ones)
        self.C = params['C']
        
        # reward scheme (still work in progress)
        self.reward_scheme = params['reward_scheme']
        if self.reward_scheme == 'mission-based':
            self.rewards_per_mission = params['reward_per_mission']
        
        # Possible actions for each aircraft 
        #    - fly one of the 5 Missions
        #    - standby
        #    - send to maintenance
        n_actions = len(self.missions)+2
        # Action space
        self.actions = list(self.missions.keys()) + ['SB'] + ['MT']
        self.action_space = spaces.Discrete(n_actions**self.n_tail)
        action_list = list(np.arange(len(self.actions)))
        # create variable with all possible actions
        self.possible_actions = np.array([p for p in itertools.product(action_list, repeat=self.n_tail)])

        # environment timestep (episode ends after N timesteps)
        self.timestep = 0
        
        # colormaps 
        self.color_groups = sns.color_palette('Dark2',len(self.group_tail)) 
        self.color_actions = list(sns.color_palette('tab10',len(self.actions)))
        # assign white color to stand-by action
        self.color_actions[len(self.missions)] = (1,1,1)  
        # assign black color to the maintenance action
        self.color_actions[len(self.missions)+1] = (0.2,0.2,0.2) 
        # colormap for discretisation
        cmap = plt.get_cmap('YlOrRd',self.dlevels)
        self.color_levels = [cmap(_) for _ in np.linspace(0,1,self.dlevels)]
        
        # create mission profile (creates self.dates and self.mission_per_date)
        self.create_mission_profile(params,verbose)
        mission_keys = ['missions_per_day','mission_composition','date_start','date_end']
        self.mission_params = {key: params[key] for key in mission_keys}

        # States (damage status, maintenance status, prescribed missions,selection)
        # initialise vector
        self.state = np.empty(2*self.n_tail+len(self.missions),dtype=int)
        # convert initial crack lengths to damage levels
        self.crack_lengths = self.a0*1000*np.ones(self.n_tail)
        damage_levels = np.searchsorted(self.dintervals, self.crack_lengths, side='right')
        # add damage levels to state vector
        self.state[:self.n_tail] = damage_levels
        # add status (all available initially)
        self.state[self.n_tail:2*self.n_tail] = np.zeros(self.n_tail,dtype=int)
        # add mission_todo from the prescribed list
        self.state[2*self.n_tail:2*self.n_tail+len(self.missions)] = self.missions_per_date[self.timestep]
        
        # initialise variables to store evaluation metrics during the episode
        self.availability = np.nan*np.ones(len(self.dates))
        self.flown_missions = list(np.zeros(self.n_tail,dtype=int))
        self.n_prev_maintenance = list(np.zeros(self.n_tail,dtype=int))
        self.n_corr_maintenance = list(np.zeros(self.n_tail,dtype=int))
        self.cumulated_damage = np.zeros(self.n_tail)
        self.cumulated_stress = np.zeros(self.n_tail)
        # other temporary variables
        self.temp_action = np.nan*np.ones([len(self.dates),self.n_tail])
        self.temp_reward = np.nan*np.ones([len(self.dates),self.n_tail]) 
        
        if verbose:
            print('Number of aircrafts: %d'%self.n_tail)
            print('Number of actions per tail number: %d'%(len(self.actions)))
            print(self.actions)
            # amax_m = self.amax*1000
            # a1,a2,a3,a4,fig = degradation_model(110,100,amax_m,
            #                                     C=self.C,levels=self.dlevels) 
            print('Mission mode: %s'%params['mission_composition'])
            print('Reward scheme: %s'%self.reward_scheme)
            print('Cyclic stress frequency: %d Hz'%self.f0)
            prev_maint_text = '%d days %d mm %d$'%(self.preventive_maintenance[0],
                                                   self.preventive_maintenance[1],
                                                   self.preventive_maintenance[2])
            prev_corr_text = '%d days %d mm %d$'%(self.corrective_maintenance[0],
                                                  self.corrective_maintenance[1],
                                                  self.corrective_maintenance[2])
            print('Preventive maintenance: %s'%prev_maint_text)
            print('Corrective maintenance: %s'%prev_corr_text)
            print('Number of possible actions: %d'%self.action_space.n)
            print('Number of possible states: %d'%((self.dlevels-1)**self.n_tail))

    def step(self, action):
        # check that action is legal
        idx_possible = get_possible_actions(self.state,self)
        if tuple(action) not in self.possible_actions[idx_possible]:
            raise Exception('Illegal action was chosen')    
            
        # read state variables
        damage, status, missions_todo = self.decode_state(self.state)
        done = False
        total_reward = 0
        
        # loop each tail number and perform the given action
        for i in range(self.n_tail):
            
            idx_action = action[i]
            action_name = self.actions[idx_action]   
            idx_tail = i
            
            # if action is to fly a mission
            if action_name in self.missions.keys():
    
                # if mission selected was not prescribed in missions_todo
                if missions_todo[idx_action] == 0:
                        raise Exception('Illegal action: fly a mission that is not prescribed')
    
                # fly the mission
                maneuvers = self.missions[action_name]
                a_initial = copy.deepcopy(self.crack_lengths[idx_tail])
                a0 = self.crack_lengths[idx_tail]/1000  
                # perform maneuvers and propagate damage
                for m,period in enumerate(maneuvers):
                    # if crack length goes above critical value
                    if a0 >= self.amax:
                        a0 = self.amax
                        break
                    # propagate crack length
                    ds = self.maneuvers[m]
                    a_new = propagate_crack(a0,period,ds,self.C[idx_tail],self.f0)
                    a0 = a_new
                
                # update crack length and damage level
                self.crack_lengths[idx_tail] = a0*1000
                damage = np.searchsorted(self.dintervals, self.crack_lengths, side='right')
                # store cumulated damage and cumulated stress
                self.cumulated_damage[idx_tail] += (self.crack_lengths[idx_tail]-a_initial)
                self.cumulated_stress[idx_tail] += self.stress_per_mission[idx_action]
                    
                # if tail number fails during operation apply corrective maintenance
                if damage[idx_tail] == self.dlevels:
                    # store how many times maintenance was applied to each tail number
                    self.n_corr_maintenance[idx_tail] += 1
                    duration = self.corrective_maintenance[0]
                    repair = self.corrective_maintenance[1]
                    cost = self.corrective_maintenance[2]
                    # update damage and status
                    a_new = np.max([self.crack_lengths[idx_tail] - repair,self.a0*1000])
                    self.crack_lengths[idx_tail] = a_new
                    status[idx_tail] = duration 
                    reward = -cost
                
                # otherwise reward the tail number for the mission flown
                else:
                    # there are 3 reward schemes
                    if self.reward_scheme == 'constant':
                        reward = 1
                    elif self.reward_scheme == 'mission-based':
                        reward = self.rewards_per_mission[idx_action]
                    elif self.reward_scheme == 'damage-based':
                        reward = (damage[idx_tail]-a_initial)*10
                    else:
                        raise Exception('Invalid reward scheme')
                    missions_todo[idx_action] -= 1
                    self.flown_missions[idx_tail] += 1
              
            # if action is to stand-by, reward = 0
            elif action_name == 'SB':
                reward = -1
                # if under maintenance, reduce the maintenance waiting period
                if status[idx_tail] > 0:
                    status[idx_tail] -= 1
            
            # if action is to apply preventive maintenance
            elif action_name == 'MT':
                # store how many times maintenance was applied to each tail number
                self.n_prev_maintenance[idx_tail] += 1
                duration = self.preventive_maintenance[0]
                repair = self.preventive_maintenance[1]
                cost = self.preventive_maintenance[2] 
                # update damage and status
                a_new = np.max([self.crack_lengths[idx_tail] - repair,self.a0*1000])
                self.crack_lengths[idx_tail] = a_new
                status[idx_tail] = duration 
                reward = -cost   
                
            else:
                raise Exception('Invalid action')
            
            # store the selected action and reward
            self.temp_action[self.timestep,idx_tail] = idx_action
            self.temp_reward[self.timestep,idx_tail] = reward
            # add reward for each tail
            total_reward += reward
    
        # update damage level
        damage = np.searchsorted(self.dintervals, self.crack_lengths, side='right')    
        # store completion rate and cumulated damage for that timestep
        self.availability[self.timestep] = 100*(1-np.sum(missions_todo)/np.sum(self.missions_per_date[self.timestep]))
        # increment timestep
        self.timestep += 1
        # if last timestep
        if self.timestep == len(self.dates):
            done = True
        else:
            # load new missions
            missions_todo = np.array(self.missions_per_date[self.timestep])
            # reduce maintenance waiting periods
            # status[status > 0] -= 1                
    
        # update state vector with new damage, status, missions_todo and selection
        self.state = self.encode_state(damage,status,missions_todo)
        
        return np.array(self.state), total_reward, done
            
    def reset(self,regenerate_missions=False):
        self.timestep = 0
        # re-initialise variables to store evaluation metrics
        self.availability = np.nan*np.ones(len(self.dates))
        self.flown_missions = list(np.zeros(self.n_tail,dtype=int))
        self.n_prev_maintenance = list(np.zeros(self.n_tail,dtype=int))
        self.n_corr_maintenance = list(np.zeros(self.n_tail,dtype=int))
        self.cumulated_damage = np.zeros(self.n_tail)
        self.cumulated_stress = np.zeros(self.n_tail)
        # other temporary variables
        self.temp_action = np.nan*np.ones([len(self.dates),self.n_tail])
        self.temp_reward = np.nan*np.ones([len(self.dates),self.n_tail]) 
        # States (damage status, maintenance status, prescribed missions,selection)
        # initialise vector
        self.state = np.empty(2*self.n_tail+len(self.missions),dtype=int)
        # convert initial crack lengths to damage levels
        self.crack_lengths = self.a0*1000*np.ones(self.n_tail)
        damage_levels = np.searchsorted(self.dintervals,self.crack_lengths,side='right')
        # add damage levels to state vector
        self.state[:self.n_tail] = damage_levels
        # add status (all available initially)
        self.state[self.n_tail:2*self.n_tail] = np.zeros(self.n_tail,dtype=int)
        # add mission_todo from the prescribed list
        self.state[2*self.n_tail:2*self.n_tail+len(self.missions)] = self.missions_per_date[self.timestep]
        if regenerate_missions:
            # create mission profile (creates self.dates and self.mission_per_date)
            self.create_mission_profile(self.mission_params)
        
    def encode_state(self,damage,status,missions_todo):
        state = np.empty(2*self.n_tail+len(self.missions),dtype=int)
        state[:self.n_tail] = damage
        state[self.n_tail:2*self.n_tail]   = status 
        state[2*self.n_tail:2*self.n_tail+len(self.missions)] = missions_todo
        
        return state
    
    def decode_state(self, state):
        damage = state[:self.n_tail]
        status = state[self.n_tail:2*self.n_tail]  
        missions_todo = state[2*self.n_tail:2*self.n_tail+len(self.missions)]        

        return damage, status, missions_todo

    def fleet_status(self):
        
        damage, status, missions_todo = self.decode_state(self.state)
        
        # setup figure
        fig,ax = plt.subplots(3,2,figsize=[14, 8],tight_layout=True,
                              gridspec_kw={'height_ratios':[3,1,2.5],
                                           'width_ratios':[6,2]})
        ax[0,0].grid(which='both',ls=':',c='0.5',lw=1)
        ax[0,0].set_xticks(np.arange(self.n_tail))
        ax[1,0].grid(which='both',ls=':',c='0.5',lw=1)
        ax[2,0].grid(which='both',ls=':',c='0.5',lw=1)

        # if crack length is discretised, show the damage levels
        if self.dlevels > 0:
            intervals = np.linspace(self.a0*1000,self.amax*1000,self.dlevels)
            for k in range(1,len(intervals)):
                ax[0,0].axhspan(ymin=intervals[k-1],ymax=intervals[k],
                                fc=self.color_levels[k],alpha=0.25)
                if k < len(intervals)-1:
                    ax[0,0].axhline(y=intervals[k],ls='-',c='k',lw=0.5,alpha=0.75)

        # plot the crack length of each tail number
        ax[0,0].plot(self.crack_lengths,'k-',lw=1)
        for i,group in enumerate(list(self.group_tail.keys())):
            ax[0,0].plot(self.group_tail[group],
                    self.crack_lengths[self.group_tail[group]],
                    'o',c=self.color_groups[i],ms=8,mec='k',label=group)

        # plot a cross if the tail number is not available
        idx_av = np.where(np.logical_or(status>0,status==-1))[0]
        ax[0,0].plot(idx_av,self.crack_lengths[idx_av]*1000,'kx')
        for k in idx_av:
            ax[0,0].text(k,self.crack_lengths[k]+1,'%d'%status[k],ha='center',fontsize=10)
        ax[0,0].axhline(y=self.a0*1000,lw=1.5,ls='--',c='g')
        ax[0,0].axhline(y=self.amax*1000,lw=1.5,ls='--',c='r')
        # ax[0,0].legend(loc='lower right',ncol=6,fontsize=10,
        #                labelspacing=0.2,columnspacing=0.2)
        ax[0,0].set(xlabel='Tail numbers',ylabel='crack length [mm]',
               title='Fleet status at timestep: %d'%self.timestep,
               ylim=[self.a0*1000-5,self.amax*1000+3])

        if self.timestep > 0:
            
            # plot mission profile for the day
            mission_profile = self.missions_per_date[self.timestep-1]
            bottom = 0
            for i in range(len(mission_profile)):
                ax[0,1].bar(0,mission_profile[i],
                       label=list(self.missions.keys())[i],
                       fc=self.color_actions[i],ec='k',bottom=bottom)
                bottom = bottom + mission_profile[i]
            ax[0,1].legend(loc='lower right',fontsize=10,framealpha=0.5)
            ax[0,1].set(xticks=[],title='%d missions'%sum(mission_profile),
                        ylim=[0,np.max(np.sum(self.missions_per_date,axis=1))])
            
            # plot Actions and Rewards
            ax[1,0].set_yscale('symlog')
            ax[1,0].axhline(y=0,ls='--',c='k')
            # ax[1,0].plot(self.temp_reward[self.timestep-1,:],'-',c='k',lw=1)
            for k in range(self.n_tail):
                action_idx = self.temp_action[self.timestep-1,k]
                if np.isnan(action_idx): 
                    continue 
                else:
                    action_idx = int(action_idx)
                # ax[1,0].plot(k,self.temp_reward[self.timestep-1,k],
                #              'o',ms=8,mec='k',mew=1,c=self.color_actions[action_idx])
                ax[1,0].bar(k,self.temp_reward[self.timestep-1,k],width=0.25,
                            fc=self.color_actions[action_idx],ec='k')
                # plot stanby action as a black cross
                if action_idx == len(self.actions)-2:
                    ax[1,0].plot(k,self.temp_reward[self.timestep-1,k],'kx',ms=10)
            ylim = [-self.corrective_maintenance[2]-1,2*np.max(self.rewards_per_mission)+1]
            if self.reward_scheme == 'constant': ylim[1] = 2
            elif self.reward_scheme == 'mission-based': ylim[1] = 6
            elif self.reward_scheme == 'damage-based': ylim[1] = 11
            title = 'Actions / Rewards at timestep %d : '%(self.timestep-1)
            title += 'total rewards = %d  -  '%np.nansum(self.temp_reward[self.timestep-1,:])
            title += 'availability = %.1f%%'%(self.availability[self.timestep-1])
            ax[1,0].set(ylabel='rewards',xlabel='Tail numbers',ylim=ylim,title=title,
                        xlim=ax[0,0].get_xlim(),xticks=np.arange(self.n_tail))
            handles = []
            for k,key in enumerate(self.actions):
                handles.append(mpatches.Patch(fc=self.color_actions[k],ec='k',label=key))
            ax[1,0].legend(handles=handles,loc='center left',fontsize=9,
                            handlelength=1,edgecolor='k',bbox_to_anchor=[1,0.5])
            
            ax[0,0].set_xticklabels([str(_) for _ in np.arange(1,self.n_tail+1)])
            ax[1,0].set_xticklabels([str(_) for _ in np.arange(1,self.n_tail+1)])

            # plot time-series
            ax[2,0].plot(self.dates[:self.timestep],np.nansum(self.temp_reward[:self.timestep,:],axis=1),'C0-',lw=1)
            title = 'Total cumulated rewards: %d  , '%np.sum(np.nansum(self.temp_reward,axis=1))
            title += 'completion rate = %.1f%%'%np.nanmean(self.availability)
            ax[2,0].set(ylabel='rewards',title=title,ylim=[-20,20])
    
            twinx = ax[2,0].twinx()
            twinx.plot(self.dates,self.availability,'C1-',ms=3,alpha=0.5)
            twinx.set(ylabel='fleet availability [%]',ylim = [-10,110])
            ax[2,0].spines['left'].set_color('C0')
            ax[2,0].spines['right'].set_color('C1')
            ax[2,0].tick_params(axis='y', colors='C0')
            ax[2,0].yaxis.label.set_color('C0')
            twinx.tick_params(axis='y', colors='C1')
            twinx.yaxis.label.set_color('C1')
            
            # plot number of missions completed
            ax[1,1].set(xlabel='Tail numbers')
            ax[1,1].grid(which='both',ls=':',lw=0.5,c='0.5')
            for i in range(self.n_tail):
                # print('Tail %d'%(i+1))
                bottom = 0
                for k,key in enumerate(list(self.actions)):
                    sum_mission = sum(self.temp_action[:,i] == k)
                    # print('%s = %03d'%(key,sum_mission))
                    ax[1,1].bar(i+1,sum_mission,bottom=bottom,
                                ec='k',fc=self.color_actions[k],alpha=0.75)
                    bottom += sum_mission
            sum_flown = np.sum(self.flown_missions)
            sum_total = np.sum(np.sum(self.missions_per_date,axis=0))
            prc = 100*sum_flown/sum_total
            ax[1,1].set(title='Total = %d (%d%%, out of %d)'%(sum_flown,prc,sum_total),)
                        #ylim=[np.min(self.flown_missions)*0.9,ax[1,1].get_ylim()[1]])
            # twinx = ax[1,1].twinx()
            # twinx.plot(self.cumulated_damage,'ko-')
            # twinx.set(yticks=[])
            
            # plot number of maintenances
            ax[2,1].set(xlabel='Tail numbers')
            ax[2,1].grid(which='both',ls=':',lw=0.5,c='0.5')
            for k in range(self.n_tail):
                ax[2,1].bar(k+1,self.n_prev_maintenance[k],ec='k',fc='C2',alpha=0.75)
                ax[2,1].bar(k+1,self.n_corr_maintenance[k],bottom=self.n_prev_maintenance[k],
                            ec='k',fc='C3',alpha=0.75)
                ax[2,1].text(k+1,0.7*self.n_prev_maintenance[k],
                             '%d'%self.n_prev_maintenance[k],ha='center')
            twinx = ax[2,1].twinx()
            twinx.plot(np.arange(1,len(self.C)+1),self.C,'ko-')
            twinx.set(yticks=[])
            ax[2,1].set(title='Number of maint. and C-values',)
                        # ylim=[np.min(self.n_prev_maintenance)*0.9,ax[2,1].get_ylim()[1]])
            handles = []
            handles.append(mpatches.Patch(fc='C2',ec='k',label='prev',alpha=0.75))
            handles.append(mpatches.Patch(fc='C3',ec='k',label='corr',alpha=0.75))
            ax[2,1].legend(handles=handles,loc='lower right',fontsize=9,
                           handlelength=1,edgecolor='k',)#bbox_to_anchor=[1,0.5])
            # set xticks
            if self.n_tail < 10: 
                ax[1,1].set_xticks(np.arange(1,self.n_tail+1))
                ax[2,1].set_xticks(np.arange(1,self.n_tail+1))
                            
        return fig
                                
    def create_mission_profile(self,params,verbose=False):
        
        # generate list of dates
        current_date = params['date_start']
        final_date = params['date_end']
        dates = []
        date = current_date
        while date < final_date:
            # exclude weekends
            if date.isoweekday() not in (6,7):
                dates.append(date)
            date = date + timedelta(days=1)  
            
        # 'mission_composition' can have different values which will determine how the prescribed missions are calculated
        
        # 'fixed', just add fix number of missions one of each type (if more missions than types duplicate them starting from the last one)
        if params['mission_composition'] == 'fixed-constant':
            n = params['missions_per_day']
            if n % len(self.missions) == 0:
                missions_per_date = int((n/len(self.missions)))*np.ones([len(dates),len(self.missions)],dtype=int)
            # in case there are more missions than mission types
            else:
                missions = np.zeros(len(self.missions),dtype=int)
                sum_missions = 0
                while sum_missions < n:
                    for i in range(len(self.missions)-1,-1,-1):
                        missions[i] +=1
                        sum_missions = np.sum(missions)
                        if sum_missions >= n:
                            break   
                missions_per_date = np.tile(missions,(len(dates),1))
        
        # 'fixed-constant-random', same as fixed constant but chooses the extra mission randomly
        elif params['mission_composition'] == 'fixed-constant-random':
            n = params['missions_per_day']
            missions_per_date = int((n/len(self.missions)))*np.ones([len(dates),len(self.missions)],dtype=int)
            if n == len(self.missions)+1:
                missions_add = np.zeros(missions_per_date.shape,dtype=int)
                for i in range(missions_add.shape[0]):
                    missions_add[i,np.random.randint(len(self.missions))] = 1
                missions_per_date += missions_add
            else:
                raise Exception('This case is not handled in %s'%params['mission_composition'])
                
        # 'fixed-mixed', fixed number of missions but fully random composition
        elif params['mission_composition'] == 'fixed-mixed':
            missions_per_date = np.zeros([len(dates),len(self.missions)],dtype=int)
            for i in range(len(dates)):
                mission_order = np.random.choice(np.arange(len(self.missions)),
                                                 len(self.missions),replace=False)
                tot_missions = 0
                for k in mission_order[:-1]:
                    n = np.random.randint(1,np.ceil(params['missions_per_day']/2)+1)
                    if (tot_missions + n) <= params['missions_per_day']:
                        tot_missions += n
                        missions_per_date[i,k] = n
                    else:
                        missions_per_date[i,k] = params['missions_per_day'] - tot_missions
                        tot_missions += params['missions_per_day'] - tot_missions
                        break
                if tot_missions < params['missions_per_day']:
                    missions_per_date[i,mission_order[-1]] = params['missions_per_day'] - tot_missions

        # 'random', each mission type drawn from specified uniform distribution (random number of missions and random composition)
        elif params['mission_composition'] == 'random': 
            missions_per_date = np.zeros([len(dates),len(self.missions)],dtype=int)
            for i in range(len(dates)):
                # randomly draw the number of missions
                n_std = params['missions_per_day']/6
                n_missions = params['missions_per_day'] + np.round(n_std*np.random.normal(size=1)[0]).astype(int)
                n_missions = np.max([1,n_missions]) # never a negative number of missions
                n_missions = np.min([n_missions,self.n_tail]) # never more missions that tail numbers
                if n_missions == 0: continue
                # mix the composition
                mission_order = np.random.choice(np.arange(len(self.missions)),
                                                 len(self.missions),replace=False)
                tot_missions = 0
                for k in mission_order[:-1]:
                    if n_missions > 1:
                        n = np.random.randint(1,np.round(n_missions/2).astype(int)+1)
                    else:
                        n = 1
                    if (tot_missions + n) <= n_missions:
                        tot_missions += n
                        missions_per_date[i,k] = n
                    else:
                        missions_per_date[i,k] = n_missions - tot_missions
                        tot_missions += n_missions - tot_missions
                        break
                if tot_missions < n_missions:
                    missions_per_date[i,mission_order[-1]] = n_missions - tot_missions
            # previous code, draws the number of each mission type from a uniform distribution 
            # missions_per_date = np.zeros([len(dates),len(self.missions)])
            # mission_range = list(np.arange(params['uniform_dist'][0],params['uniform_dist'][1]+1))
            # for i in range(len(dates)):
                # random.seed(i)
                # missions_per_date[i,:] = random.sample(mission_range,len(self.missions))
        
        # manually choose the mission composition
        elif params['mission_composition'] == 'custom':
            missions = params['custom_missions']
            missions_per_date = np.tile(missions,(len(dates),1))
            
        else:
            raise Exception('invalid value for mission_composition in params')
        
        # create environment variables
        self.dates = dates
        self.missions_per_date = missions_per_date
        
        if verbose:
            print('%d missions generated from %s to %s (%d days)'%(np.sum(self.missions_per_date),
                                                                  self.dates[0].strftime('%d-%m-%Y'),
                                                                  self.dates[-1].strftime('%d-%m-%Y'),
                                                                  len(self.dates)))
            # make figure
            if params['mission_composition'] in ['fixed-constant','fixed-constant-random','fixed-mixed','custom']: 
                fig,ax = plt.subplots(1,1,figsize=[15, 4],tight_layout=True)
                ax.grid(which='both',ls=':',c='0.5',lw=1)
                idx = np.arange(20)
                dates_idx = [dates[_] for _ in idx]
                bottom = np.zeros(len(idx))
                for i in range(len(self.missions)):
                    ax.bar(dates_idx,missions_per_date[idx,i],
                           label=list(self.missions.keys())[i],
                           fc=self.color_actions[i],ec='k',bottom=bottom)
                    bottom = bottom + missions_per_date[idx,i]
                ax.legend(loc='upper right')
                ax.set(title='First 20 days of missions (continues to %s)'%self.dates[-1].strftime('%d-%m-%Y'),
                          ylabel='number of missions')
            else:
                fig,ax = plt.subplots(2,1,figsize=[15, 6],tight_layout=True)
                ax[0].grid(which='both',ls=':',c='0.5',lw=1)
                idx = np.arange(20)
                dates_idx = [dates[_] for _ in idx]
                bottom = np.zeros(len(idx))
                for i in range(len(self.missions)):
                    ax[0].bar(dates_idx,missions_per_date[idx,i],
                           label=list(self.missions.keys())[i],
                           fc=self.color_actions[i],ec='k',bottom=bottom)
                    bottom = bottom + missions_per_date[idx,i]
                ax[0].legend(loc='upper right')
                ax[0].set(title='First 20 days of missions (continues to %s)'%self.dates[-1].strftime('%d-%m-%Y'),
                          ylabel='number of missions')
                bins = np.arange(np.min(np.sum(missions_per_date,axis=1)),
                                 np.max(np.sum(missions_per_date,axis=1))+2,
                                 1)-0.5
                if len(bins) == 1: bins = 10
                ax[1].hist(np.sum(missions_per_date,axis=1),ec='k',alpha=0.5,
                           bins=bins)
                ax[1].set(title='Distribution of the number of missions per day',
                          xlabel='number of missions per day',ylabel='counts')
            return fig
    
    def update_fleet(self, past_missions, verbose=False):
        damage, status, missions_todo, selection = self.decode_state(self.state)
        mission_types = list(self.missions.keys())
        # update the fleet state with the past missions
        for group in self.group_tail.keys():
            missions_todo = past_missions[group]
            # print('%s completed: '%group,end='')
            # for each tail number in the group
            for i in self.group_tail[group]:
                # get current crack length
                a0 = self.crack_lengths[i]/1000
                for k,number in enumerate(missions_todo):
                    # if this mission was performed
                    if number > 0:
                        # select the mission and perform it N times
                        action = self.missions[mission_types[k]]
                        for l in range(number):
                            # each mission consists a sequence of maneuvers
                            for m,period in enumerate(action):
                                if a0 >= self.amax:
                                    a0 = self.amax
                                    break
                                ds = self.maneuvers[m]
                                a_new = propagate_crack(a0,period,ds,self.C[i],self.f0)
                                a0 = a_new
                        # if i == self.group_tail[group][0]:
                            # print('%d M%d'%(number,k+1), end=', ')
                self.crack_lengths[i] = a0*1000
                # set max crack length just below the critical value
                if self.crack_lengths[i] == self.amax*1000: self.crack_lengths[i] -= 1
            damage_levels = np.searchsorted(self.dintervals,self.crack_lengths,side='right')
            self.state[:self.n_tail] = damage_levels
            status[damage_levels==self.dlevels] = -1
            self.state[self.n_tail:2*self.n_tail] = status
            
            # print the state of the fleet
            # print('')
            # if verbose:
            #     [print('%d: %.4f'%(_,damage[_])) for _ in self.group_tail[group]]
        
        if verbose:
            damage, status, missions_todo, selection = self.decode_state(self.state)
            # plot the crack length of each tail number
            fig,ax = plt.subplots(1,1,figsize=[8, 5],tight_layout=True,)    
            ax.grid(which='both',ls=':',c='0.5',lw=1)
            ax.plot(self.crack_lengths,'k-',lw=1)
            for i,group in enumerate(list(self.group_tail.keys())):
                ax.plot(self.group_tail[group],
                        self.crack_lengths[self.group_tail[group]],
                        'o',c=self.color_groups[i],ms=10,mec='k',label=group)
            # plot a cross if the tail number is not available
            idx_av = np.where(np.logical_or(status>0,status==-1))[0]
            ax.plot(idx_av,self.crack_lengths[idx_av]*1000,'kx')
            for k in idx_av:
                ax.text(k,self.crack_lengths[k]+1,'%d'%status[k],ha='center',fontsize=10)
            ax.axhline(y=self.a0*1000,lw=1.5,ls='--',c='g')
            ax.axhline(y=self.amax*1000,lw=1.5,ls='--',c='r')
            ax.legend(loc='lower right',ncol=6,fontsize=10,
                           labelspacing=0.2,columnspacing=0.2)
            ax.set(xlabel='Tail numbers',ylabel='crack length [mm]',
                   title='Fleet status at timestep: %d'%self.timestep,
                   ylim=[self.a0*1000-5,self.amax*1000+3]) 
            if self.dlevels > 0:
                intervals = np.linspace(self.a0*1000,self.amax*1000,self.dlevels)
                for k in range(1,len(intervals)):
                    ax.axhspan(ymin=intervals[k-1],ymax=intervals[k],
                                    fc=self.color_levels[k],alpha=0.25)
                    if k < len(intervals)-1:
                        ax.axhline(y=intervals[k],ls='-',c='k',lw=0.5,alpha=0.75)

#%% 3. Auxiliary functions

def get_possible_actions(state,env):
    "get index of all possible actions for a given state (to avoid illegal actions)"
    # start_time = time.time()
    damage, status, missions_todo = env.decode_state(state)
    # initialise variables
    idx_possible = np.ones(len(env.possible_actions),dtype=(bool))
    # 2 steps to filter out the action combinations that are not possible:
    # Step 1: if a tail is under maintenance, only keep combinations where that tail stands-by  
    standby_action = len(env.missions)
    if np.any(status > 0):
        idx_status = np.where(status>0)[0]
        for i in idx_status:
            # for each tail find the possible action combinations
            idx = env.possible_actions[:,i] == standby_action
            idx_possible[~idx] = False
    # keep track of the indices
    idx_dummy = np.where(idx_possible)[0]
    possible_actions_dummy = env.possible_actions[idx_possible,:]
    # Step 2: fly a mission that was not prescribed
    for i in range(len(env.missions)):
        # for each presecribed mission find the possible action combinations
        idx = ~(np.sum(possible_actions_dummy == i,axis=1) > missions_todo[i])
        idx_possible[idx_dummy[~idx]] = False
    # print("--- %s seconds ---" % (time.time() - start_time))
    return np.where(idx_possible)[0]

def episode_qtable(q_table,env,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with RL policy from q-table"
    
    # nested function to read from the q-table
    def q(state, action=None):
        # only the damage levels go in the state
        state = str(state[:env.n_tail])
        # if state is not present in the table, add a vector of nans
        if state not in q_table:
            q_table[state] = np.nan*np.ones(len(env.possible_actions))
        # if action is not specified return all q-values for that state
        if action is None:
            return q_table[state]
        # otherwise return q-value for specified action
        else:
            idx_action = np.where([np.all(action == _) for _ in env.possible_actions])[0][0]
            qval = q_table[state][idx_action]
            # if q-value is a nan, return 0 instead
            if np.isnan(qval): qval = 0
            return qval
    
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_qtable')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    state_qval = {'states':[],'qvals':[]}
    done = False
    while not done:
        current_state = copy.deepcopy(env.state)
        damage, status, missions_todo, = env.decode_state(current_state)
        idx_possible = get_possible_actions(env.state,env)
        
        ###############################################################
        # # remove standby actions
        # idx_SB = np.where([_ == 'SB' for _ in env.actions])[0][0]
        # idx_remove = []
        # for l in np.where(status == 0)[0]:
        #     idx_remove += list(np.where(env.possible_actions[idx_possible,l] == idx_SB)[0])
        # if len(idx_remove) > 0:
        #     idx_remove = np.unique(idx_remove)
        #     idx_possible = np.delete(idx_possible,idx_remove)
        ###############################################################

        # select argmax action amongst possible actions
        if str(current_state[:env.n_tail]) in q_table:
            predicted = q(current_state)
            predicted = predicted[idx_possible]
            # if all nans, then choose a random action
            if not np.all(np.isnan(predicted)):
                actions_possible = env.possible_actions[idx_possible,:]
                action = actions_possible[np.nanargmax(predicted)]
            else:
                action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
        else:
            action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]       
        # store state-qvalues pairs
        state_qval['states'].append(current_state)
        predicted = q(current_state)
        qvals = np.nan*np.ones(len(predicted))
        for k in idx_possible: qvals[k] = predicted[k]
        state_qval['qvals'].append(qvals)
        new_state, reward, done = env.step(action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save a figure for the first N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts, state_qval

def episode_random(env,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with a random policy"
    # whether to plot a few steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_random')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    done = False
    while not done:
        damage, status, missions_todo, = env.decode_state(env.state)
        idx_possible = get_possible_actions(env.state,env)
        # choose a random action
        random_action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
        new_state, reward, done = env.step(random_action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save a figure for the first N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts

def episode_on_condition(env,maint_level=4,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with a random mission assignment and on-condition maintenance"
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_oncond')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    standby_action = len(env.missions)
    maint_action = len(env.missions) + 1
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    done = False
    while not done:
        damage, status, missions_todo, = env.decode_state(env.state)
        idx_possible = get_possible_actions(env.state,env)
        actions_possible = env.possible_actions[idx_possible,:]
        # for on-condition maintenance, first remove all action combinations with maintenance
        actions_possible = actions_possible[~np.any(actions_possible == maint_action,axis=1)]
        # then remove all stand-by actions unless a tail number is under maintenance
        if np.any(status == 0):
            idx_status = np.where(status==0)[0]
            # if there are more tails than missions, put a random one in standby
            if len(idx_status) > np.sum(missions_todo):
                idx_random = np.random.choice(np.arange(env.n_tail),1,replace=False)[0]
                idx_status = np.delete(idx_status,idx_random)
            for i in idx_status:
                # for each tail find the possible action combinations
                idx = actions_possible[:,i] == standby_action
                actions_possible = actions_possible[~idx]
        # choose a random action amongst the left ones
        random_action = actions_possible[np.random.choice(np.arange(len(actions_possible)),1,replace=False)[0]]
        # now, look at the damage thresholdand apply maintenance accordingly
        idx_unav = np.where(status>0)[0]
        for i in range(env.n_tail):
            if damage[i] >= maint_level and i not in idx_unav:
                random_action[i] = maint_action
        # step
        new_state, reward, done = env.step(random_action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save a figure for N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts

def episode_force_life(env,maint_level=4,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with force-life management and on-condition maintenance"   
    
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_forcelife')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    standby_action = len(env.missions)
    maint_action = len(env.missions) + 1
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    done = False
    while not done:
        damage, status, missions_todo, = env.decode_state(env.state)
        # accumulate damage
        cumulative_damage = env.cumulated_damage
        # rank tails by how damaged they are relative to others
        idx_sorted = np.argsort(cumulative_damage)
        # remove the tails that are not available
        idx_unav = np.where(status>0)[0]
        if len(idx_unav) > 0:
            idx_remove = []
            for k in idx_unav:
                idx_remove.append(np.where(idx_sorted == k)[0][0])
            idx_sorted = np.delete(idx_sorted,idx_remove)
        # assign hardest mission to most damage tail number
        prescribed_missions = copy.deepcopy(missions_todo)
        actions = standby_action*np.ones(env.n_tail,dtype=int)
        mission_difficulty = np.array(env.rewards_per_mission) - 1
        counter = 0
        for k in np.argsort(mission_difficulty)[::-1]:
            while prescribed_missions[k] > 0 and counter < len(idx_sorted):
                actions[idx_sorted[counter]] = k
                # print('tail #%d -> action %d'%(idx_sorted[counter],k))
                prescribed_missions[k] -= 1
                counter += 1
        # overwrite with on-condition maintenance
        for i in range(env.n_tail):
            if damage[i] >= maint_level and i not in idx_unav: 
                actions[i] = maint_action
        new_state, reward, done = env.step(actions)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save figure for N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts

def episode_equal_stress(env,maint_level=4,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with equal stress management and on-condition maintenance" 
    
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_equal_stress')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    standby_action = len(env.missions)
    maint_action = len(env.missions) + 1
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    done = False
    while not done:
        damage, status, missions_todo, = env.decode_state(env.state)
        # accumulate damage
        cumulative_damage = env.cumulated_stress
        # rank tails by how damaged they are relative to others
        idx_sorted = np.argsort(cumulative_damage)
        # remove the tails that are not available
        idx_unav = np.where(status>0)[0]
        if len(idx_unav) > 0:
            idx_remove = []
            for k in idx_unav:
                idx_remove.append(np.where(idx_sorted == k)[0][0])
            idx_sorted = np.delete(idx_sorted,idx_remove)
        # assign hardest mission to most damage tail number
        prescribed_missions = copy.deepcopy(missions_todo)
        actions = standby_action*np.ones(env.n_tail,dtype=int)
        mission_difficulty = np.array(env.rewards_per_mission) - 1
        counter = 0
        for k in np.argsort(mission_difficulty)[::-1]:
            while prescribed_missions[k] > 0 and counter < len(idx_sorted):
                actions[idx_sorted[counter]] = k
                # print('tail #%d -> action %d'%(idx_sorted[counter],k))
                prescribed_missions[k] -= 1
                counter += 1
        # overwrite with on-condition maintenance
        for i in range(env.n_tail):
            if damage[i] >= maint_level and i not in idx_unav:
                actions[i] = maint_action
        new_state, reward, done = env.step(actions)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save figure for N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts

def episode_model(model,env,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with RL model from NN weights"
    
    # whether to plot a few steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_NN')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    state_qval = {'states':[],'qvals':[]}
    done = False
    while not done:
        current_state = copy.deepcopy(env.state)
        damage_state = current_state[:env.n_tail]
        damage, status, missions_todo, = env.decode_state(current_state)
        idx_possible = get_possible_actions(env.state,env)
        actions_possible = env.possible_actions[idx_possible,:]
        predicted = model.predict(damage_state).detach().numpy()
        predicted = predicted[idx_possible]
        action = actions_possible[np.nanargmax(predicted)]
        new_state, reward, done = env.step(action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # store state-q-values pairs
        # state_qval['states'].append(damage_state)
        # qvals = np.nan*np.ones(len(predicted))
        # for k in idx_possible: qvals[k] = predicted[k]
        # state_qval['qvals'].append(qvals)
        # select argmax action amongst possible actions
        # predicted = predicted[idx_possible]
        # action = actions_possible[np.argmax(predicted)]
        
        # save figure for N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts, state_qval

def episode_qtable_split(q_table,env,env_half,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with RL policy from q-table"
    
    # nested function to read from the q-table (linked to env_half!!!)
    def q(state, action=None):
        # only the damage levels go in the state
        state = str(state[:env_half.n_tail])
        # if state is not present in the table, add a vector of nans
        if state not in q_table:
            q_table[state] = np.nan*np.ones(len(env_half.possible_actions))
        # if action is not specified return all q-values for that state
        if action is None:
            return q_table[state]
        # otherwise return q-value for specified action
        else:
            idx_action = np.where([np.all(action == _) for _ in env_half.possible_actions])[0][0]
            qval = q_table[state][idx_action]
            # if q-value is a nan, return 0 instead
            if np.isnan(qval): qval = 0
            return qval
    
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_qtable')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    state_qval = {'states':[],'qvals':[]}
    done = False
    idx_split = int(env.n_tail/2)
    while not done:
        current_state = copy.deepcopy(env.state)
        damage, status, missions_todo, = env.decode_state(current_state)
        # split actions in 2 sets
        damage1 = damage[:idx_split]
        damage2 = damage[idx_split:]
        status1 = status[:idx_split]
        status2 = status[idx_split:]
        
        # split the missions
        missions_todo1 = np.zeros(len(missions_todo),dtype=int)
        # find one that is double
        missions_todo1[np.where(missions_todo == 2)[0][0]] = 1
        idx_missions_todo1 = np.random.choice(np.where(missions_todo == 1)[0],idx_split-1,replace=False)
        for k in idx_missions_todo1: missions_todo1[k] = 1
        missions_todo2 = missions_todo - missions_todo1

        # randomly split the missions in 2 sets
        # missions_todo1 = np.zeros(len(missions_todo),dtype=int)
        # missions_todo_copy = copy.deepcopy(missions_todo)
        # idx_missions = np.random.choice(len(missions_todo),len(missions_todo),replace=False)
        # while sum(missions_todo1) < idx_split:
        #     for idx in idx_missions:
        #         if missions_todo_copy[idx] > 0:
        #             m = np.random.randint(0,missions_todo_copy[idx]+1)
        #             if sum(missions_todo1) + m <= idx_split:
        #                 missions_todo1[idx] += m
        #                 missions_todo_copy[idx] -= m
        #             else:
        #                 break
        # missions_todo2 = missions_todo - missions_todo1
        
        # print(missions_todo)
        # print(missions_todo1)
        # print(missions_todo2)
        if not np.all((missions_todo1 + missions_todo2) == missions_todo):
            raise Exception('Mission splitting went wrong.')
            
        # get possible actions for each set
        state1 = env_half.encode_state(damage1,status1,missions_todo1)
        idx_possible1 = get_possible_actions(state1,env_half)
                
        # uncomment to check things are right
        # predicted1 = q(state1)
        # for k in idx_possible1:
        #     print('%s -> %d'%(env_half.possible_actions[k],predicted1[k]))
        # predicted1 = predicted1[idx_possible1]
        # action_max = env_half.possible_actions[idx_possible1[np.argmax(predicted1)]]
        # print('MAX: %s --> %d'%(action_max,np.max(predicted1)))
        
        state2 = env_half.encode_state(damage2,status2,missions_todo2)
        idx_possible2 = get_possible_actions(state2,env_half)
        
        ###############################################################
        # remove standby actions
        # idx_SB = np.where([_ == 'SB' for _ in env_half.actions])[0][0]
        # idx_remove = []
        # for l in np.where(status1 == 0)[0]:
        #     idx_remove += list(np.where(env_half.possible_actions[idx_possible1,l] == idx_SB)[0])
        # if len(idx_remove) > 0:
        #     idx_remove = np.unique(idx_remove)
        #     idx_possible1 = np.delete(idx_possible1,idx_remove) 
        # idx_remove = []
        # for l in np.where(status2 == 0)[0]:
        #     idx_remove += list(np.where(env_half.possible_actions[idx_possible2,l] == idx_SB)[0])
        # if len(idx_remove) > 0:
        #     idx_remove = np.unique(idx_remove)
        #     idx_possible2 = np.delete(idx_possible2,idx_remove) 
        ###############################################################

        # find best action for each set
        if str(damage1) in q_table:
            predicted1 = q(state1)
            predicted1 = predicted1[idx_possible1]
            if not np.all(np.isnan(predicted1)):
                actions_possible1 = env_half.possible_actions[idx_possible1,:]
                action1 = actions_possible1[np.nanargmax(predicted1)]
            else:
                action1 = env_half.possible_actions[np.random.choice(idx_possible1,1,replace=False)[0]]
        else:
            action1 = env_half.possible_actions[np.random.choice(idx_possible1,1,replace=False)[0]]   
        if str(damage2) in q_table:
            predicted2 = q(damage2)
            predicted2 = predicted2[idx_possible2]
            if not np.all(np.isnan(predicted2)):
                actions_possible2 = env_half.possible_actions[idx_possible2,:]
                action2 = actions_possible2[np.nanargmax(predicted2)]
            else:
                action2 = env_half.possible_actions[np.random.choice(idx_possible2,1,replace=False)[0]]
        else:
            action2 = env_half.possible_actions[np.random.choice(idx_possible2,1,replace=False)[0]]   

        # combine actions from each set
        action = np.array(list(action1)+list(action2))

        # step             
        new_state, reward, done = env.step(action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save a figure for the first N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts, state_qval

def episode_qtable_split2(q_table,env,env_half,plot_steps=False,N=150,fp=os.getcwd()):
    "run a full episode with RL policy from q-table"
    
    # nested function to read from the q-table
    def q(state, action=None):
        # only the damage levels go in the state
        state = str(state[:env.n_tail])
        # if state is not present in the table, add a vector of nans
        if state not in q_table:
            q_table[state] = np.nan*np.ones(len(env.possible_actions))
        # if action is not specified return all q-values for that state
        if action is None:
            return q_table[state]
        # otherwise return q-value for specified action
        else:
            idx_action = np.where([np.all(action == _) for _ in env.possible_actions])[0][0]
            qval = q_table[state][idx_action]
            # if q-value is a nan, return 0 instead
            if np.isnan(qval): qval = 0
            return qval
    
    # whether to plot some steps
    if plot_steps:
        fp_steps = os.path.join(fp,'steps_qtable')
        if not os.path.exists(fp_steps): os.makedirs(fp_steps)
        plt.ioff()  
    damage_ts = np.nan*np.ones((len(env.dates),env.n_tail))
    state_qval = {'states':[],'qvals':[]}
    done = False
    idx_split = int(env.n_tail/2)
    while not done:
        current_state = copy.deepcopy(env.state)
        damage, status, missions_todo, = env.decode_state(current_state)
        
        # split state in 2 sets
        damage1 = damage[:idx_split]
        damage2 = damage[idx_split:]
        status1 = status[:idx_split]
        status2 = status[idx_split:]
        missions_todo1 = [1,2]
        missions_todo2 = [1,2]
                        
        # get possible actions for each set
        state1 = env_half.encode_state(damage1,status1,missions_todo1)
        idx_possible1 = get_possible_actions(state1,env_half)
        state2 = env_half.encode_state(damage2,status2,missions_todo2)
        idx_possible2 = get_possible_actions(state2,env_half)
        
        # find best action for each set
        if str(damage1) in q_table:
            predicted1 = q(damage1)
            predicted1 = predicted1[idx_possible1]
            if not np.all(np.isnan(predicted1)):
                actions_possible1 = env_half.possible_actions[idx_possible1,:]
                action1 = actions_possible1[np.nanargmax(predicted1)]
            else:
                action1 = env_half.possible_actions[np.random.choice(idx_possible1,1,replace=False)[0]]
        else:
            action1 = env_half.possible_actions[np.random.choice(idx_possible1,1,replace=False)[0]]   
        if str(damage2) in q_table:
            predicted2 = q(damage2)
            predicted2 = predicted2[idx_possible2]
            if not np.all(np.isnan(predicted2)):
                actions_possible2 = env_half.possible_actions[idx_possible2,:]
                action2 = actions_possible2[np.nanargmax(predicted2)]
            else:
                action2 = env_half.possible_actions[np.random.choice(idx_possible2,1,replace=False)[0]]
        else:
            action2 = env_half.possible_actions[np.random.choice(idx_possible2,1,replace=False)[0]]   

        # combine actions from each set
        action = np.array(list(action1)+list(action2))

        # step             
        new_state, reward, done = env.step(action)
        damage_ts[env.timestep-1,:] = env.crack_lengths
        
        # save a figure for the first N steps
        if plot_steps:
            if env.timestep > 0 and env.timestep <= N:
                fig = env.fleet_status()
                fig.savefig(os.path.join(fp_steps,'status_%04d.jpg'%env.timestep))
                plt.close(fig)
    reward_ts = copy.deepcopy(env.temp_reward)
    plt.ion()
    return damage_ts, reward_ts, state_qval

#%% 4. Plotting functions

def plot_episode(damage_ts,reward_ts,env):
    "plot the damage and reward evolution during an episode"
        
    fig,ax = plt.subplots(2,3,figsize=[17, 9],tight_layout=True,
                          gridspec_kw={'height_ratios':[1,1],
                                       'width_ratios':[2,1,1]})
    ax[1,0].grid(which='both',ls=':',c='0.5',lw=1)
    colors = env.color_groups
    width = 1/(env.n_tail+1)

    # plot discretised levels
    if env.dlevels > 0:
        intervals = np.linspace(env.a0*1000,env.amax*1000,env.dlevels)
        intervals2 = np.arange(env.dlevels)+0.5
        for k in range(0,len(intervals)):
            ax[0,0].axhspan(ymin=intervals[k-1],ymax=intervals[k],
                            fc=env.color_levels[k],alpha=0.25)
            ax[0,1].axhspan(ymin=intervals2[k-1],ymax=intervals2[k],
                            fc=env.color_levels[k],alpha=0.25)
            ax[0,2].axhspan(ymin=intervals2[k-1],ymax=intervals2[k],
                            fc=env.color_levels[k],alpha=0.25)
            if k < len(intervals):
                ax[0,0].axhline(y=intervals[k],ls='-',c='k',lw=0.5,alpha=0.75)
                ax[0,1].axhline(y=intervals2[k],ls='-',c='k',lw=0.5,alpha=0.75)
                ax[0,2].axhline(y=intervals2[k],ls='-',c='k',lw=0.5,alpha=0.75)
    # plot damage [0,0]
    damage_levels = np.searchsorted(env.dintervals,damage_ts,side='right')
    ax[0,0].set(xlabel='time',ylabel='damage level',title='Damage history')
    for i in range(damage_ts.shape[1]):
        ax[0,0].plot(env.dates,damage_ts[:,i],'-',color=colors[i],label='tail #%d'%(i+1))
        idx_maint = np.where(env.temp_action[:,i] == len(env.actions)-1)[0]-1
        for k in idx_maint: 
            ax[0,0].plot(env.dates[k],damage_ts[k,i],'o',color=colors[i])
    ax[0,0].set_yticks(intervals[:-1]+np.diff(intervals)[0]/2)
    ax[0,0].set_yticklabels(['%d'%_ for _ in np.arange(1,env.dlevels)])
    ax[0,1].set_yticks(np.arange(1,env.dlevels))
    ax[0,1].set_yticklabels(['%d'%_ for _ in np.arange(1,env.dlevels)])
    ax[0,2].set_yticks(np.arange(1,env.dlevels))
    ax[0,2].set_yticklabels(['%d'%_ for _ in np.arange(1,env.dlevels)])
    ax[0,0].set_ylim([intervals[0],intervals[-1]])
    ax[0,1].set_ylim([0.5,env.dlevels-0.5])
    ax[0,2].set_ylim([0.5,env.dlevels-0.5])
    ax[0,1].set_title('Maint. levels')
    ax[0,2].set_title('Time at each damage level')
    # plot maintenance levels for each tail [0,1]
    for i in range(env.n_tail):
        idx_maint = np.where(env.temp_action[:,i] == len(env.actions)-1)[0]-1
        dl = damage_levels[idx_maint,i]
        for d in np.unique(dl):
            ax[0,1].barh(y=d+width*(i-np.ceil((env.n_tail/2-1))),width=np.sum(dl==d),
                         fc=colors[i],ec='k',height=width)
    # plot histogram of damage levels [0,2]
    for i in range(env.n_tail):
        dl = damage_levels[:,i]
        for d in np.unique(dl):
            ax[0,2].barh(y=d+width*(i-np.ceil((env.n_tail/2-1))),width=np.sum(dl==d),
                         fc=colors[i],ec='k',height=width)
    # add legends
    handles = []
    for k in range(env.n_tail):
        handles.append(mpatches.Patch(fc=env.color_groups[k],ec='k',label='tail %d'%(k+1)))
    ax[0,0].legend(handles=handles,loc='center',fontsize=11,ncol=3,
              handlelength=1,edgecolor='k')
    handles = []
    for i in range(env.n_tail):
        count_maint = sum(env.temp_action[:,i] == len(env.actions)-1)
        handles.append(mpatches.Patch(fc=env.color_groups[i],ec='k',label='%d'%count_maint))
    ax[0,1].legend(handles=handles,loc='center right',fontsize=11,
              handlelength=1,edgecolor='k')
    
    # plot reward [1,0]
    total_rewards = np.sum(np.nansum(env.temp_reward,axis=1))
    if not total_rewards == 0.0:
        for i in range(reward_ts.shape[1]):
            ax[1,0].plot(env.dates,np.nancumsum(reward_ts[:,i]),'-',ms=4,
                         color=colors[i],
                         label='%d%%'%(100*np.nansum(reward_ts[:,i])/total_rewards));
    title = 'Total cumulated rewards: %d'%total_rewards
    ax[1,0].set(ylabel='rewards',title=title)
    ax[1,0].legend(loc='lower right',handlelength=1,markerscale=2)
    inset = ax[1,0].inset_axes([0.075,0.65,0.3,0.3])
    standbys = env.temp_action == len(env.actions)-2
    standbys = np.sum(standbys, axis=1)
    for k in range(env.n_tail+1):
        inset.bar(x=k,height=100*np.sum(standbys==k)/len(standbys),
                  width=0.5,fc='0.5',ec='k') 
    inset.set(xlabel='simul maint')
    # plot number of missions completed [1,1]
    for i in range(env.n_tail):
        bottom = 0
        for k,key in enumerate(list(env.actions)):
            sum_mission = sum(env.temp_action[:,i] == k)
            ax[1,1].bar(i+1,sum_mission,bottom=bottom,
                        ec='k',fc=env.color_actions[k],alpha=0.75)
            ax[1,1].text(i+1,bottom+sum_mission/2,
                         '%d%%'%(100*sum_mission/len(env.dates)),
                         va='center',ha='center',fontsize=11)
            bottom += sum_mission
    sum_flown = np.sum(env.flown_missions)
    sum_total = np.sum(np.sum(env.missions_per_date,axis=0))
    prc = 100*sum_flown/sum_total
    ax[1,1].set(title='%d out of %d (%d%%)'%(sum_flown,sum_total,prc),
                xlabel='Tail numbers',yticks=[])
    # availability, C-values, cumulated damage [1,2]
    ax[1,2].set(xlabel='Tail numbers',ylabel='Cumulated damage [mm]',
                title='C-values and Cumulated damage')
    for i in range(env.n_tail):
        ax[1,2].bar(x=i+1,height=env.cumulated_damage[i],ec='k',
                    fc=env.color_groups[i],alpha=0.75)
    twinx = ax[1,2].twinx()
    twinx.plot(np.arange(1,env.n_tail+1),env.C,'--o',mfc='w',c='0.25')
    twinx.set(ylabel='C-values')
    twinx.tick_params(axis='y', colors='0.25')
    twinx.yaxis.label.set_color('0.25') 
    
    return fig

def plot_damage_and_reward(damage_ts,reward_ts,env):
    "plot the damage and reward evolution during an episode"
    # setup figure
    fig,ax = plt.subplots(2,2,figsize=[14, 8],tight_layout=True,
                          gridspec_kw={'height_ratios':[1,1],
                                       'width_ratios':[4,1]})
    ax[0,0].grid(which='both',ls=':',c='0.5',lw=1)
    ax[0,1].grid(which='both',ls=':',c='0.5',lw=1)
    ax[1,0].grid(which='both',ls=':',c='0.5',lw=1)
    ax[1,1].grid(which='both',ls=':',c='0.5',lw=1)
    colors = sns.color_palette('Set2',env.n_tail) 
    
    # plot discretised levels
    if env.dlevels > 0:
        intervals = np.linspace(env.a0*1000,env.amax*1000,env.dlevels)
        for k in range(1,len(intervals)):
            ax[0,0].axhspan(ymin=intervals[k-1],ymax=intervals[k],
                            fc=env.color_levels[k],alpha=0.25)
            if k < len(intervals)-1:
                ax[0,0].axhline(y=intervals[k],ls='-',c='k',lw=0.5,alpha=0.75)

    # plot damage [0,0]
    ax[0,0].set(xlabel='time',ylabel='crack length [mm]',title='Damage history')
    for i in range(damage_ts.shape[1]):
        ax[0,0].plot(env.dates,damage_ts[:,i],'o-',color=colors[i],label='tail #%d'%(i+1));
    # ax[0,0].set(ylim=[env.a0*1000-1,env.amax*1000+1])
    ax[0,0].axhline(y=env.amax*1000,ls='--',c='k')
    # plot reward [1,0]
    ax[1,0].set(xlabel='time',ylabel='reward',title='Cumulated Rewards')
    for i in range(reward_ts.shape[1]):
        ax[1,0].plot(env.dates,np.nancumsum(reward_ts[:,i]),'o-',ms=4,color=colors[i],label='tail #%d'%(i+1));
    # ax[1,0].legend(loc='upper left')
    # plot C values [0,1]
    ax[0,1].set(xlabel='tail number',ylabel='Cumulated damage [mm]',
                title='Cumulated damage')
    ax[0,1].plot(env.cumulated_damage,'k-')
    for i in range(len(env.cumulated_damage)):
        ax[0,1].plot(i,env.cumulated_damage[i],'o',mfc=colors[i],mec='k',ms=10)
    # ax[0,1].set_ylim([np.nanmean(env.cumulated_damage)*0.5,np.nanmean(env.cumulated_damage)*1.5])
    twinx = ax[0,1].twinx()
    twinx.plot(np.arange(env.n_tail),env.cumulated_stress/1e9,'--o',mfc='w',c='0.5')
    # twinx.set(ylim=[np.nanmean(env.cumulated_stress/1e9)*0.5,np.nanmean(env.cumulated_stress/1e9)*1.5],
              # ylabel='cumulated stress range')
    twinx.tick_params(axis='y', colors='0.5')
    twinx.yaxis.label.set_color('0.5')
    # plot cumulated rewards [1,1]
    ax[1,1].set(xlabel='tail number',ylabel='total reward',title='Total reward per tail number')
    for i in range(reward_ts.shape[1]):
        ax[1,1].bar(x=i,height=np.nansum(reward_ts[:,i]),fc=colors[i],label='tail #%d'%(i+1),ec='k');
    ax[1,1].set(ylim=[np.min(np.nansum(reward_ts,axis=0))-30,np.max(np.nansum(reward_ts,axis=0))+10])
    twinx = ax[1,1].twinx()
    twinx.plot(env.C,'--o',mfc='w',c='0.5')
    twinx.set(ylabel='C-values')
    twinx.tick_params(axis='y', colors='0.5')
    twinx.yaxis.label.set_color('0.5')    
    ax[0,0].set_title('episode total rewards = %d'%np.nansum(reward_ts))
    
    return fig

def plot_eps_alpha(train_params):
    "plot the epsilon and alpha curves"
    n_episodes = train_params['n_episodes']
    max_epsilon = train_params['max_epsilon'] 
    min_epsilon = train_params['min_epsilon']
    epsilon_decay = train_params['decay_epsilon']
    max_alpha = train_params['max_alpha']
    min_alpha = train_params['min_alpha']  
    alpha_decay = train_params['decay_alpha']
    episodes = np.arange(0,n_episodes,1)
    epsilons = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episodes)
    alphas = min_alpha + (max_alpha - min_alpha) * np.exp(-alpha_decay * episodes)
    fig,ax = plt.subplots(1,1,figsize=(8,6),tight_layout=True,sharex=True)
    ax.grid(which='both',ls=':',c='0.5',lw=1)
    ax.set(title= r'$\epsilon$' + '-greedy strategy and learning rate ' + r'$\alpha$' + ' over %d episodes'%n_episodes,
            xlabel='number of episodes')
    ax.plot(episodes,epsilons,'-',lw=2,label=r'$\epsilon$' + ' (max=%.1f, min=%.3f)'%(np.max(epsilons),np.min(epsilons)))
    ax.plot(episodes,alphas,'-',lw=2,label=r'$\alpha$'+ ' (max=%.1f, min=%.3f)'%(np.max(alphas),np.min(alphas)))
    ax.legend(loc='upper right')
    return fig, epsilons, alphas

def plot_qvals(q_table,env,reward_ts):
    "plot the qvalues as a function of damage level for each tail"
    fig,ax = plt.subplots(1,1,figsize=(10,8),tight_layout=True,)
    ax.grid(which='major',ls=':',c='k',lw=1)
    ax.set(title='Policy map - total reward = %d'%(np.nansum(reward_ts)),
           xlabel='tail #1',xticks=np.arange(1,env.dlevels),
           ylabel='tail #2',yticks=np.arange(1,env.dlevels))
    ax.axis('equal')
    for state in q_table:
        state_arr = np.fromstring(state[1:-1],dtype=int, sep=' ')
        best_action = np.nanargmax(q_table[state])
        action_pair = env.possible_actions[best_action]
        ax.plot(state_arr[0],state_arr[1],marker=11,color='C%d'%action_pair[0],
                   ms=25,mec='k')
        ax.plot(state_arr[0],state_arr[1],marker=10,color='C%d'%action_pair[1],
                   ms=25,mec='k')
    handles = []
    for k,key in enumerate(env.actions):
        handles.append(mpatches.Patch(fc='C%d'%k,ec='k',label=key))
    ax.legend(handles=handles,loc='center left',fontsize=14,
              handlelength=1,edgecolor='k',bbox_to_anchor=[1,0.5],
              title='Actions')
    return fig

def plot_qvals3(q_table,env):
    "plot the qvalues as a function of damage level for each tail when 3 tail numbers"
    from mpl_toolkits import mplot3d
    
    # plot Q-table
    colors = sns.color_palette('Blues',len(env.actions)-2)
    color_actions = []
    for k in range(len(env.actions)-2):
        color_actions.append(colors[k])
    colors = sns.color_palette('tab10',10)
    color_actions.append(colors[2])
    color_actions.append(colors[3])
    handles = []
    for k,key in enumerate(env.actions):
        handles.append(mpatches.Patch(fc=color_actions[k],ec='k',label=key))
    states = np.arange(1,10)
    
    figs = []
    for k in range(env.n_tail):
        fig = plt.figure(figsize=(14,10),tight_layout=True)
        ax = plt.axes(projection='3d')
        ax.set(xlabel='damage level tail #1',
               ylabel='damage level tail #2',
               zlabel='damage level tail #3',
               title = 'Preferred action for tail #%d'%(k+1))
        for i,s in enumerate(states):
            for ii,ss in enumerate(states):
                for iii,sss in enumerate(states): 
                    state = '[%d %d %d]'%(s,ss,sss)
                    if not state in q_table: continue
                    if np.all(np.isnan(q_table[state])): continue
                    best_action = np.nanargmax(q_table[state])
                    action_pair = env.possible_actions[best_action]
                    ax.scatter3D(s,ss,sss,s=50,color=color_actions[action_pair[k]],
                                 ec='k',linewidth=0.5)
    
        ax.legend(handles=handles,loc='center left',fontsize=14,
                  handlelength=1,edgecolor='k',bbox_to_anchor=[1,0.5],
                  title='Actions')
        figs.append(fig)
    return figs
                    
def plot_agent(model):
    "plot the model weights"
    weights = list(model.parameters())
    n_layers = int(len(weights)/2)
    fig,ax = plt.subplots(n_layers,2,figsize=(8,6),tight_layout=True)
    for axt in ax: 
        for axtt in axt:
            axtt.grid(which='major',c='0.5',ls=':',lw=0.5)
    for k in range(n_layers):
        layer_weights = weights[k*2].cpu().detach().numpy()
        layer_biases = weights[2*k+1].cpu().detach().numpy()
        ax[k,0].hist(layer_weights.flatten(),ec='k')
        ax[k,0].set_title('Weights: Dense layer ( %s x %s )'%(weights[k*2].shape[0],
                                                            weights[k*2].shape[1]))
        ax[k,1].hist(layer_biases.flatten(),ec='k',align='left')
        ax[k,1].set_title('Biases ( %d x 1 )'%len(weights[2*k+1])) 
    return fig

def coverage_qtable(q_table,env):
    "calculate coverage of the q-table based on remaining nan values"
    
    n_actions = len(env.possible_actions)
    # calculate percentage coverage of the actions in each state
    coverage_table = q_table.copy()
    for key in coverage_table.keys():
        coverage_table[key] = 100*(1-sum(np.isnan(q_table[key]))/n_actions)
    mean_coverage = np.mean(list(coverage_table.values()))
    # print('Average coverage %.1f%%'%mean_coverage)
    # make figure
    fig,axs = plt.subplots(1,env.n_tail,figsize=[17,5],tight_layout=True) 
    if env.n_tail == 1:
        axs = [axs]
    for i,ax in enumerate(axs): 
        ax.grid(which='major',ls=':',c='0.5',lw=0.5)
        ax.set(title='Tail #%d'%(i+1), ylim=[0,100],xlabel='damage levels')
    axs[0].set_ylabel('% actions explored')
    axs[0].set_title('Tail #1 - average coverage %.1f%%'%mean_coverage)
    n_actions = len(env.possible_actions)
    states = list(q_table.keys())
    states = np.array([[int(x) for x in _.strip('[]').split(' ')] for _ in states])
    for i in range(env.n_tail):
        for j in range(1,env.dlevels):
            idx_state = np.where(states[:,i] == j)[0]
            prc_nans = [coverage_table[str(states[_,:])] for _ in idx_state]
            bp = axs[i].boxplot(prc_nans,sym='+',positions=[j],
                                widths=0.65,patch_artist=True,showfliers=True,
                                whiskerprops={'linewidth':1},
                                flierprops={'alpha':0.3,'markersize':4,
                                            'color':'0.5'})
            for median in bp['medians']:
                median.set(color='k', linewidth=1)
            for boxes in bp['boxes']:
                boxes.set(facecolor=env.color_levels[j-1],alpha=0.75,)
            # add median value as text
            axs[i].text(j,np.median(prc_nans)+0.05,
                        '%.1f'%np.median(prc_nans),
                        horizontalalignment='center', fontsize=10)
    return coverage_table, fig
    
#%% 5. Reinforcement Learning with Q-table

def train_qtable(env,q_table,train_params,fp,reward_per_episode=[]):
    "perform Q-learning to update look-up table"
    
    plt.ioff()
    
    # nested function to read from the q-table
    def q(state, action=None):
        # only the damage levels go in the state
        state = str(state[:env.n_tail])
        # if state is not present in the table, add a vector of nans
        if state not in q_table:
            q_table[state] = np.nan*np.ones(len(env.possible_actions))
        # if action is not specified return all q-values for that state
        if action is None:
            return q_table[state]
        # otherwise return q-value for specified action
        else:
            idx_action = np.where([np.all(action == _) for _ in env.possible_actions])[0][0]
            qval = q_table[state][idx_action]
            # if q-value is a nan, return 0 instead
            if np.isnan(qval): qval = 0
            return qval
    
    # read params dictionary
    n_episodes = train_params['n_episodes']
    gamma = train_params['gamma']
    crack_lengths = train_params['crack_lengths']
    step = train_params['saving_step']
    if 'n_decisions' in train_params.keys():
        n_decisions = train_params['n_decisions']
    else:
        n_decisions = len(env.dates)
    # search parameters
    fig, epsilons, alphas = plot_eps_alpha(train_params)
    plt.close(fig)
    
    # start search
    if reward_per_episode:
        print('using loaded rewards')
    else:
        reward_per_episode = []
    maint_per_episode = []
    count_ep = 0
    best_reward = -np.inf
    for n in range(n_episodes):
        
        # print('\repisode %d/%d'%(n+1,n_episodes),end='')
        
        total_reward = 0
        done = False  
        
        # select search parameters
        epsilon = epsilons[n]
        alpha = alphas[n]
        
        # initial conditions
        env.reset(regenerate_missions=True)
        # random initial conditions
        env.crack_lengths = copy.deepcopy(crack_lengths[n,:])
        damage_levels = np.searchsorted(env.dintervals,env.crack_lengths,side='right')
        env.state[:env.n_tail] = damage_levels    

        # run episode
        while not done:
            current_state = copy.deepcopy(env.state)
            damage, status, missions_todo = env.decode_state(env.state)
            # get possible actions
            idx_possible = get_possible_actions(current_state,env)
            actions_possible = env.possible_actions[idx_possible,:]
            # draw random number
            r = np.random.rand()
            # either explore a new action
            if r < epsilon:
                action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
            # or exploit best known action
            else:
                if str(current_state[:env.n_tail]) in q_table:
                    predicted = q(current_state)
                    predicted = predicted[idx_possible]
                    if not np.all(np.isnan(predicted)):
                        action = actions_possible[np.nanargmax(predicted)]
                    else:
                        action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
                else:
                    action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
            # execute action and calculate reward and new state
            new_state, reward, done = env.step(action)
            total_reward += reward
            # bellman's update
            if env.timestep < len(env.dates) and not np.all(np.isnan(q(new_state))):
                max_future_q = reward + gamma*np.nanmax(q(new_state))
            else:
                max_future_q = reward
            action_idx =  np.where([np.all(action == _) for _ in env.possible_actions])[0][0]               
            q(current_state)[action_idx] = q(current_state, action) + alpha*(max_future_q - q(current_state, action))
            
            # stop the episode after n_decisions
            if env.timestep >= n_decisions: 
                done = True
        # print metrics for user
        count_ep += 1
        if not count_ep % step == 0: continue
        
        # evaluate current model by running M episodes and calculating the average reward
        sum_rewards = 0
        average_maint = np.zeros(env.n_tail)
        for k in range(train_params['repetitions']):
            # start_time = time.time()
            env.reset(True)
            # initialise state
            env.crack_lengths = np.random.uniform(env.a0*1000,env.amax*1000,env.n_tail)
            damage_levels = np.searchsorted(env.dintervals,env.crack_lengths,side='right')
            env.state[:env.n_tail] = damage_levels
            damage_ts, reward_ts, qvals = episode_qtable(q_table,env)  
            # print("--- %.5f seconds ---" % (time.time() - start_time))
            # print('qtable -> total rewards = %d'%np.nansum(reward_ts))
            sum_rewards += np.nansum(reward_ts)
            average_maint += np.array(env.n_prev_maintenance)
        average_reward = sum_rewards/train_params['repetitions']
        average_maint = average_maint/train_params['repetitions']

        # append to time-series of reward
        reward_per_episode.append(average_reward)
        maint_per_episode.append(average_maint)
        
        # plot loss and reward time-series
        window = 50
        fig,ax = plt.subplots(1,1,figsize=(10,5),tight_layout=True)
        ax.grid(which='major',ls=':',c='k',lw=1)
        indices = (np.arange(len(reward_per_episode))+1)*step-1	
        idx_max = indices[np.argmax(reward_per_episode)]
        ax.plot(indices,reward_per_episode,'C0o-',ms=3)
        ax.set(title='Reward per episode - max = %d - current %d'%(np.nanmax(reward_per_episode),
                                                                   reward_per_episode[-1]),
               xlabel='episodes',ylabel='total reward')
        ax.plot(idx_max,np.nanmax(reward_per_episode),'ko',mfc='none')
        if len(reward_per_episode) > window:
            ma = np.convolve(reward_per_episode, np.ones(window), 'valid') / window	
            ax.plot(indices[window-1:],ma,'r-',lw=2,label='%d-moving average'%window)	
            ax.legend(loc='lower right')
        ax2 = ax.twinx()
        ax2.plot(indices,np.mean(np.array(maint_per_episode),axis=1),'C1-o',ms=3)
        fig.savefig(os.path.join(fp,'loss_temp.jpg')) 
        plt.close(fig)
        
        # if it is an improvement, save the model
        if average_reward > best_reward:
            best_reward = average_reward
            # store Q-table
            with open(os.path.join(fp,'qtable_best.pkl'),'wb') as f:
                pickle.dump(q_table,f) 
            # plot episode
            env.reset()
            damage_ts, reward_ts, qvals = episode_qtable(q_table,env)
            fig = plot_episode(damage_ts,reward_ts,env)
            fig.savefig(os.path.join(fp,'episode_best.jpg'))
            plt.close(fig)
            # store rewards
            train_stats = {'reward':reward_per_episode,'maint':np.array(maint_per_episode)}
            with open(os.path.join(fp,'train_stats.pkl'),'wb') as f:
                pickle.dump(train_stats,f)
            # plot coverage
            coverage_table, fig = coverage_qtable(q_table, env)
            fig.savefig(os.path.join(fp,'coverage_best.jpg'))
            plt.close(fig)
        # otherwise plot temp figures
        else:
            # plot episode
            fig = plot_episode(damage_ts,reward_ts,env)
            fig.savefig(os.path.join(fp,'episode_temp.jpg'))
            plt.close(fig)
            # plot coverage
            coverage_table, fig = coverage_qtable(q_table, env)
            fig.savefig(os.path.join(fp,'coverage_temp.jpg'))
            plt.close(fig)

        print('\r%d - rew %d (best %d) - cov %.1f%% - eps %.3f - alpha = %.3f'%(n+1,average_reward,best_reward,np.mean(list(coverage_table.values())),epsilon,alpha),end='')

        
    # store training stats from full RL search
    fig,ax = plt.subplots(1,1,figsize=(10,8),tight_layout=True)
    ax.grid(which='major',ls=':',c='k',lw=1)
    ax.plot(np.arange(len(reward_per_episode)),reward_per_episode,'C0o-',ms=4)
    ax.set(title='total reward per episode %d (max = %d)'%(reward_per_episode[-1],np.nanmax(reward_per_episode)),
           xlabel='episodes')
    ax.plot(np.nanargmax(reward_per_episode),np.nanmax(reward_per_episode),'ko',mfc='none')
    train_stats = {'reward':reward_per_episode}
    with open(os.path.join(fp,'train_stats.pkl'),'wb') as f:
        pickle.dump(train_stats,f)
        
    # store the last Qtable (not best)
    with open(os.path.join(fp,'qtable_last.pkl'),'wb') as f:
        pickle.dump(q_table,f)
    
    return q_table

#%% 6. Reinforcement Learning with Deep Q-learning

class Net(torch.nn.Module):
    'Create a Neural Network class.'
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.005):
        super(Net, self).__init__()
        hidden_dim2 = int(hidden_dim/2)
        hidden_dim3 = int(hidden_dim/4)
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim2, hidden_dim3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim3, action_dim)
                        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, amsgrad=False)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1)
        self.criterion = torch.nn.MSELoss() 
        # self.criterion = torch.nn.SmoothL1Loss()
        
    def update(self, state, y):
            'Update the weights of the network given a training sample.'
            y_pred = self.model(torch.Tensor(state))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss
        
    def validate(self, state, y):
            'Calculate validation loss using current model weights'
            with torch.no_grad():
                y_pred = self.model(torch.Tensor(state))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
            return loss
            
    def predict(self, state):
            'Compute Q values for all actions using the Neural Net.' 
            with torch.no_grad():
                return self.model(torch.Tensor(state))

def train_agent(env,params,model,train_params,fp):
    "perform Q-learning"
    plt.ioff()
    # create folders to store outputs
    fp_batch = os.path.join(fp,'episodes')
    fp_loss = os.path.join(fp,'ts')
    fp_policy = os.path.join(fp,'policy')
    fp_models = os.path.join(fp,'models')
    if not os.path.exists(fp_batch): os.makedirs(fp_batch)
    if not os.path.exists(fp_loss): os.makedirs(fp_loss)
    if not os.path.exists(fp_policy): os.makedirs(fp_policy)
    if not os.path.exists(fp_models): os.makedirs(fp_models)
    
    # read params dictionary
    n_episodes = train_params['n_episodes']
    gamma = train_params['gamma']
    crack_lengths = train_params['crack_lengths']
    step = train_params['saving_step']
    # search parameters
    fig, epsilons, alphas = plot_eps_alpha(train_params)
    plt.close(fig)
    
    # NN update parameters
    N = train_params['update_freq']
    batch_size = train_params['batch_size']
    replay_size = train_params['replay_size']    
    # check if double Q-learning is selected
    if not 'double_QL' in train_params.keys():
        double_QL = False
    else:
        # create second model with same weights
        double_QL = train_params['double_QL']
    if double_QL:
        target_freq = train_params['target_freq']
        target_model = Net(env.n_tail,len(env.possible_actions),
                            train_params['hidden_layers'][0],
                            train_params['learning_rate'])
        target_model.load_state_dict(model.state_dict())
        print('Double Q-learning activated (update freq = %d)'%target_freq)

    # initialise variables
    replay_memory = deque(maxlen=replay_size) 
    loss_per_fit, reward_per_episode = [],[]
    count_ep , count_step = 0, 0
    env = Env(params)
    # loop through episodes
    for n in range(n_episodes):
        
        print('\repisode %d/%d'%(n+1,n_episodes),end='')
        
        total_rew = 0
        done = False  
        update_counter = 0
        
        # select search parameters
        epsilon = epsilons[n]
        alpha = alphas[n]
        
        # initial conditions
        env.reset()
        # random initial conditions
        env.crack_lengths = copy.deepcopy(crack_lengths[n,:])
        damage_levels = np.searchsorted(env.dintervals,env.crack_lengths,side='right')
        env.state[:env.n_tail] = damage_levels   
        
        # run episode
        while not done:
            current_state = copy.deepcopy(env.state)
            damage_state = current_state[:env.n_tail]
            damage, status, missions_todo = env.decode_state(env.state)
            # get possible actions
            actions_possible, idx_possible = get_possible_actions(current_state,env)
            # draw random number
            r = np.random.rand()
            # either explore a new action
            if r < epsilon:
                action = env.possible_actions[np.random.choice(idx_possible,1,replace=False)[0]]
            # or exploit best known action
            else:
                predicted = model.predict(damage_state).detach().numpy()
                predicted = predicted[idx_possible]
                action = actions_possible[np.nanargmax(predicted)]
            # execute action and calculate reward and new state
            new_state, reward, done = env.step(action)
            # store state-action-reward-newstate in buffer
            replay_memory.append([current_state,action,reward,new_state])
            # accumulate total reward for the episode
            total_rew += reward
            # increment counter
            count_step += 1
            # every N steps/decisions, train the model
            if count_step % N == 0:
                
                # randomly sample a batch from the replay buffer
                if len(replay_memory) > batch_size:
                    batch = random.sample(replay_memory, batch_size)
                else:
                    batch = replay_memory
                    
                # format batch
                states = np.array([_[0] for _ in batch])
                new_states = np.array([_[3] for _ in batch])
                # actions = np.array([_[1] for _ in batch])
                # rewards = np.array([_[2] for _ in batch])
                damage_states = states[:,:env.n_tail]
                damage_new_states = new_states[:,:env.n_tail]
                # predict on batch (if double_QL use target model)
                qs_vals = model.predict(damage_states).detach().numpy()
                if double_QL:
                    qs_vals_future = target_model.predict(damage_new_states).detach().numpy()
                else:
                    qs_vals_future = model.predict(damage_new_states).detach().numpy()

                # apply Bellman's equation to update Qvalues (output layer of model)
                X,Y = [],[]
                for index, (observation, action, reward, new_observation) in enumerate(batch):
                    action_idx = np.where([_ == action for _ in env.possible_actions])[0][0]
                    # update qs value with Bellman's equation
                    current_qs = qs_vals[index]
                    # filter illegal actions from future qvals
                    actions_possible, idx_possible = get_possible_actions(current_state,env)
                    max_future_q = reward + gamma * np.max(qs_vals_future[index][idx_possible])
                    current_qs[action_idx] = (1-alpha)*current_qs[action_idx] + alpha*max_future_q
                    # store state and udated Q-value (only damage levels go in the input layer)
                    damage_state = observation[:env.n_tail]
                    X.append(damage_state)
                    Y.append(current_qs)
                # fit the model with  batch
                loss = model.update(np.array(X),np.array(Y))
                # store loss and q-values
                loss_per_fit.append(float(loss)) 
                # print('fit at timestep %d - loss %.3f - %d samples'%(env.timestep,history.history['loss'][0],len(batch)))
                
                # if double QL update target model
                if double_QL:
                    update_counter += 1
                    if update_counter == target_freq:
                        target_model.load_state_dict(model.state_dict())
                        update_counter = 0
           
        # print metrics for user
        count_ep += 1
        if not count_ep % step == 0: continue
    
        # print('rew %d - eps %.3f - alpha = %.3f'%(total_rew,epsilon,alpha),end='')
          
        # save model
        torch.save(model,os.path.join(fp_models, 'agent_NN_%02d.pkl'%n))
        
        # evaluate current model by running one episode
        env.reset() 
        damage_ts, reward_ts, qvals = episode_model(model,env)
        plt.ioff()
        # append total reward
        reward_per_episode.append(np.nansum(reward_ts))
        
        # plot fleet status at last timestep
        fig = env.fleet_status()
        fig.savefig(os.path.join(fp_batch,'policy_%d.jpg'%n))
        plt.close(fig)
        
        # plot loss and reward time-series
        fig,ax = plt.subplots(2,1,figsize=(10,8),tight_layout=True)
        ax[0].grid(which='major',ls=':',c='k',lw=1)
        ax[1].grid(which='major',ls=':',c='k',lw=1)
        ax[0].plot(np.arange(len(loss_per_fit)),loss_per_fit,'C0-')
        ax[0].set(title='Loss per update',xlabel='updates',ylabel='loss')
        ax[1].plot(np.arange(len(reward_per_episode)),reward_per_episode,'C1-')
        ax[1].set(title='total reward per episode %d'%reward_per_episode[-1],xlabel='episodes')
        # fig.savefig(os.path.join(fp_loss,'loss_%d.jpg'%(n))) 
        fig.savefig(os.path.join(fp,'temp_loss.jpg')) 
        plt.close(fig)
        
    # store stats from RL search
    train_stats = {'loss':loss_per_fit,
                   'reward':reward_per_episode,
                  }
    with open(os.path.join(fp,'train_stats.pkl'),'wb') as f:
        pickle.dump(train_stats,f)
        
    # load best model
    idx_max = (np.argmax(train_stats['reward'])+1)*step-1
    with open(os.path.join(fp_models,'agent_NN_%02d.pkl'%idx_max),'rb') as f:
        model = pickle.load(f)
        
    # plot rewards per episode and loss
    fig,ax = plt.subplots(2,1,figsize=(10,8),tight_layout=True)
    ax[0].grid(which='major',ls=':',c='k',lw=1)
    ax[0].plot(np.arange(len(reward_per_episode)),reward_per_episode,'C0o-',ms=4)
    ax[0].set(title='total reward per episode %d (max = %d)'%(reward_per_episode[-1],np.nanmax(reward_per_episode)),
              xlabel='episodes')
    ax[0].plot(np.nanargmax(reward_per_episode),np.nanmax(reward_per_episode),'ko',mfc='none')
    ax[1].plot(np.arange(len(loss_per_fit)),loss_per_fit,'C1o-',ms=4)
    ax[1].set(title='loss per fit %d (min = %.3f)'%(loss_per_fit[-1],np.nanmin(loss_per_fit)),
              xlabel='updates')
    ax[1].plot(np.nanargmin(loss_per_fit),np.nanmin(loss_per_fit),'ko',mfc='none')
        
    # evaluate current model by running one episode
    env.reset() 
    damage_ts, reward_ts, qvals = episode_model(model,env)
    plt.ioff()
    print('model -> total rewards = %d'%np.nansum(reward_ts))
    # plot fleet status at last timestep
    fig = env.fleet_status()
    fig.savefig(os.path.join(fp,'status_best.jpg'))
    
    return model
    
#%% deprecated

# def plot_reward_simul(reward_dist,env):
#     "plot the distribution of rewards"
#     N = reward_dist.shape[0]
#     binwidth = 10
#     total_rewards = np.sum(reward_dist,axis=1)
#     colors = sns.color_palette('Set2',env.n_tail) 
#     fig,ax = plt.subplots(1,2,figsize=[14,6],tight_layout=True,gridspec_kw={'width_ratios':[2,1]})
#     ax[0].grid(which='major',ls=':',c='0.5'); ax[1].grid(which='major',ls=':',c='0.5');
#     ax[0].set(xlabel='total episode reward',ylabel='count',title='Rewards distribution for %d runs'%N,)#xlim=[1125,1275])
#     ax[1].set(xlabel='tail number',ylabel='median reward per episode',title='Reward per tail number')
#     bins = np.arange(np.min(total_rewards),np.max(total_rewards)+binwidth,binwidth)-binwidth/2
#     ax[0].hist(total_rewards,bins=bins,fc='C0',ec='k')
#     ax[0].text(np.max(total_rewards),ax[0].get_ylim()[1]/2,
#                'max %d'%np.max(total_rewards),ha='center',fontsize=12,
#                bbox=dict(fc='none', ec='black', boxstyle='round,pad=.5'))
#     ax[0].text(np.min(total_rewards),ax[0].get_ylim()[1]/2,
#                'min %d'%np.min(total_rewards),ha='center',fontsize=12,
#                bbox=dict(fc='none', ec='black', boxstyle='round,pad=.5'))
#     ax[0].text(np.mean(total_rewards),ax[0].get_ylim()[1]*0.95,
#                'mean %d'%np.mean(total_rewards),ha='center',fontsize=12,
#                bbox=dict(fc='none', ec='black', boxstyle='round,pad=.5'))
#     for i in range(reward_dist.shape[1]):
#         errors = np.zeros((2,1))
#         errors[0] = np.median(reward_dist[:,i])-np.percentile(reward_dist[:,i],10)
#         errors[1] = np.percentile(reward_dist[:,i],90) - np.median(reward_dist[:,i])
#         ax[1].bar(x=i,height=np.median(reward_dist[:,i]),yerr=np.array(errors),fc=colors[i],label='tail #%d'%(i+1),capsize=5,ec='k');
#     ax[1].set(ylim=[np.min(reward_dist)*0.9,ax[1].get_ylim()[1]])
#     twinx = ax[1].twinx()
#     twinx.plot(env.C,'ko:',mfc='w',lw=0.75,label='C values')
#     twinx.legend(loc='upper left')
    
#     return fig

# def plot_mean_qvals(mean_qvals,env):
#     "plot the evolution of the average q-value for each action during training"
#     fig,ax = plt.subplots(1,2,figsize=(10,5),tight_layout=True,
#                           gridspec_kw={'width_ratios':[4,1]})
#     ax[0].grid(which='major',ls=':',c='k',lw=1)
#     mean_qvals = np.array(mean_qvals)
#     argmax = np.argmax(mean_qvals,axis=1)
#     handles = []
#     for k,key in enumerate(env.actions):
#         ax[0].plot(mean_qvals[:,k],'-',c='C%d'%k)
#         ax[1].bar(x=k,height=sum(argmax==k),fc=env.color_actions[k],ec='k')
#         handles.append(mpatches.Patch(fc=env.color_actions[k],ec='k',label=key))
#     ax[0].set(xlabel='model updates',ylabel='mean q-value')
#     ax[1].set(ylabel='counts',xlabel='argmax action',xticks=np.arange(mean_qvals.shape[1]))
#     ax[1].legend(handles=handles,loc='center left',fontsize=9,
#                     handlelength=1,edgecolor='k',bbox_to_anchor=[1,0.5])
#     return fig

# def plot_qvals_boxplot(state_qval,rewards,env):
#     # initialise figures
#     fig,ax = plt.subplots(env.n_tail,1,figsize=(16,10),sharey=True,
#                           sharex=True,gridspec_kw=dict(top=0.95,bottom=0.05,
#                                                        left=0.08,right=0.95,
#                                                        hspace=0.1))
#     for i,tail in enumerate(list(state_qval.keys())):
#         ax[i].grid(which='major',ls=':',c='k',lw=1)
#         # read damage levels and convert from list to arrays
#         states_all = state_qval[tail]['states']
#         damage_levels = np.empty(len(states_all))
#         qvals = np.empty([len(states_all),len(env.actions)])
#         for k in range(len(states_all)):
#             damage_levels[k] = states_all[k][int(tail)]
#             qvals[k,:] = state_qval[tail]['qvals'][k]
#         for l in range(1,env.dlevels):
#             pos = l
#             idx_damage = np.where(damage_levels == l)[0]
#             for k,key in enumerate(env.actions):
#                 ax[i].boxplot(qvals[idx_damage,k],positions=[pos+k/8],widths=[0.1],
#                               showfliers=True,patch_artist=True,
#                               boxprops=dict(facecolor='C'+str(k),color='C'+str(k),alpha=0.75),
#                               medianprops=dict(color='C'+str(k),lw=1),
#                               whiskerprops=dict(color='C'+str(k),lw=0.5),
#                               capprops=dict(color='C'+str(k),lw=0.5),
#                               flierprops=dict(color='C'+str(k), markeredgecolor='C'+str(k),
#                                               marker='+',markersize=3),
#                               labels=None)
#             ax[i].set(ylabel='tail %s'%tail)
#             if i == len(state_qval)-1:
#                 ax[i].set(xlim=[0.5,env.dlevels+0.5],
#                           xlabel='damage level',xticks=np.arange(1,env.dlevels),
#                           xticklabels=[str(_) for _ in np.arange(1,env.dlevels)])  
#     ax[0].set(title='Qvals - total reward = %d (%1.f%%)'%(np.nansum(rewards),np.mean(env.availability)))
#     handles = []
#     for k,key in enumerate(env.actions):
#         handles.append(mpatches.Patch(fc='C%d'%k,ec='k',label=key))
#     ax[0].legend(handles=handles,loc='upper right',fontsize=10,
#               handlelength=1,edgecolor='k')
#     return fig

# def plot_policy(state_qval,rewards,env):
#     # initialise figures
#     fig,ax = plt.subplots(1,2,figsize=(16,7),tight_layout=True,
#                           gridspec_kw={'width_ratios':[5,1]})
#     ax[0].grid(which='major',ls=':',c='k',lw=1)
#     ax[0].set(title='Preferred action per damage level - total reward = %d (%1.f%%)'%(np.nansum(rewards),np.mean(env.availability)),
#               xlim=[0.5,env.dlevels+0.5],
#            xlabel='damage level',xticks=np.arange(1,env.dlevels),
#            ylabel='tail number',yticks=np.arange(env.n_tail))
#     for tail in state_qval.keys():
#         # read damage levels and  from list to arrays
#         states_all = state_qval[tail]['states']
#         damage_levels = np.empty(len(states_all))
#         qvals = np.empty([len(states_all),len(env.actions)])
#         for k in range(len(states_all)):
#             damage_levels[k] = states_all[k][int(tail)]
#             qvals[k,:] = state_qval[tail]['qvals'][k]
#         # calculate average q-value for each action and each damage level
#         qvals_per_level = np.empty([len(env.actions),env.dlevels-1])
#         for k,key in enumerate(env.actions):
#             qval_per_action = []
#             for l in range(1,env.dlevels):
#                 idx_damage = np.where(damage_levels == l)[0]
#                 # if that state is explored by the policy
#                 if len(idx_damage) > 0:
#                     qval_per_action.append(np.mean(qvals[idx_damage,k]))
#                 else: # otherwise put a nan (not explored)
#                     qval_per_action.append(np.nan)
#             qvals_per_level[k,:] = np.array(qval_per_action) 
#         # for each state, plot action with the highest average q-value
#         for l in range(1,env.dlevels):
#             if not np.all(np.isnan(qvals_per_level[:,l-1])):    
#                 action_max = np.argmax(qvals_per_level[:,l-1])
#                 ax[0].plot(l,int(tail),'o',mfc='C%d'%action_max,mec='k',ms=16)   
#     handles = []
#     for k,key in enumerate(env.actions):
#         handles.append(mpatches.Patch(fc='C%d'%k,ec='k',label=key))
#     ax[0].legend(handles=handles,loc='center left',fontsize=14,
#               handlelength=1,edgecolor='k',bbox_to_anchor=[1,0.5])
#     ax[0].invert_yaxis()
#     # plot c-values
#     ax[1].grid(which='major',ls=':',c='k',lw=1)
#     ax[1].barh(np.arange(env.n_tail),np.nansum(rewards,axis=0),0.5,fc='0.75',ec='k')
#     ax[1].set(xlim=[0.95*np.nanmin(np.nansum(rewards,axis=0)),
#                     1.05*np.nanmax(np.nansum(rewards,axis=0))])
#     twinx = ax[1].twiny()
#     twinx.plot(env.C,np.arange(env.n_tail),'ko-',mfc='w',ms=10)
#     twinx.invert_yaxis()
#     return fig