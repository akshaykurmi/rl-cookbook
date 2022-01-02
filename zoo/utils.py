import argparse
import os
import shutil
from time import sleep

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Train or evaluate the agent?')
    parser.add_argument('--resume', default=False, required=False, action='store_true', help='Resume training run?')
    return parser.parse_args()


def get_output_dirs(base_dir, run_id, args, dirs):
    dirs = [os.path.join(base_dir, d, run_id) for d in dirs]
    if args.mode == 'train' and args.resume is False:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    return dirs


def evaluate_policy(env, policy):
    observation = env.reset()
    env.render()
    done = False
    while not done:
        sleep(0.005)
        action = policy.sample(tf.expand_dims(observation, axis=0)).numpy()[0]
        observation, reward, done, info = env.step(action)
        env.render()
    env.close()
