import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from copy import deepcopy
import utils
from random import choice
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC

class ADAR(SAC):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        self.batch_size = args.batch_size
        self.total_steps = args.train_steps
        self.single_steps = args.single_steps
        self.alter_steps = self.total_steps / self.single_steps

    def update_critic(self, replay_buffer, obs, action, reward, next_obs, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        weak_augmentations = {
            'random_translate': augmentations.random_translate,
            'random_shift': augmentations.random_shift,
            'random_pepper': augmentations.random_pepper,
            'random_grayscale': augmentations.random_grayscale,
        }

        hard_augmentations = {
            'random_conv': augmentations.random_conv,
            'random_cutout': augmentations.random_cutout,
            'random_color_jitter': augmentations.random_color_jitter,
            'random_cutout_color': augmentations.random_cutout_color,
            'random_overlay': augmentations.random_overlay,
            'random_blur': augmentations.random_blur,
        }

        obs_ = obs / 255.0

        if self.alter_steps % 2 == 0:
            stage = self.alter_steps // 10
            augmentation = choice(list(weak_augmentations.keys()))
            obs_aug = weak_augmentations[augmentation](obs_.clone(), stage) * 255.0
        else:
            stage = self.alter_steps // 10
            augmentation = choice(list(hard_augmentations.keys()))
            obs_aug = hard_augmentations[augmentation](obs_.clone(), stage) * 255.0

        if step % self.single_steps == 0:
            self.alter_steps -= 1

        current_Q1, current_Q2 = self.critic(obs_aug, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if L is not None:
            L.log('train/critic_loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.__sample__()

        self.update_critic(replay_buffer, obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()
