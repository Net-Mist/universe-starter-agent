from collections import namedtuple
import queue
import scipy.signal
import threading
from models import *
from pseudocount import PC
import math
import numpy as np
import matplotlib.pyplot as plt


def current_lr(t: int, max_t: int, initial_lr: float, final_lr: float) -> float:
    """
    Compute and return the current learning rate
    :param t: time step
    :param max_t: time step after then learning rate doesn't decrease anymore
    :param initial_lr: initial learning rate
    :param final_lr: final learning rate
    :return: the current learning rate
    """

    if max_t == 0 or initial_lr == 0:
        return final_lr  # use fix LR

    if t <= max_t:
        return math.exp((math.log(initial_lr) - math.log(final_lr)) * (max_t - t) / max_t + math.log(final_lr))
    else:
        return final_lr


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0):
    """
given a rollout, compute its returns and the advantage
"""
    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    rewards = np.asarray(rollout.rewards)
    vpred_t = np.asarray(rollout.values + [rollout.r])

    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])
    batch_r = discount(rewards_plus_v, gamma)[:-1]
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    # this formula for the advantage comes "Generalized Advantage Estimation":
    # https://arxiv.org/abs/1506.02438
    batch_adv = discount(delta_t, gamma * lambda_)

    features = rollout.features[0]
    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout(object):
    """
a piece of a complete rollout.  We run our agent, and process its experience
once it has processed enough steps.
"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []

    def add(self, state, action, reward, value, terminal, features):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)


class RunnerThread(threading.Thread):
    """
One of the key distinctions between a normal environment and a universe environment
is that a universe environment is _real time_.  This means that there should be a thread
that would constantly interact with the environment and tell it what to do.  This thread is here.
"""

    def __init__(self, env, policy, num_local_steps, visualise, brain, a3cp):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5)
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.brain = brain
        self.a3cp = a3cp

    def start_runner(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()

    def run(self):
        with self.sess.as_default():
            self._run()

    def _run(self):
        rollout_provider = env_runner(self.env, self.policy, self.num_local_steps, self.summary_writer, self.visualise,
                                      self.brain, self.a3cp)
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number.  This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)


def env_runner(env, policy, num_local_steps, summary_writer, render, brain, a3cp):
    """
The logic of the thread runner.  In brief, it constantly keeps on running
the policy, and as long as the rollout exceeds a certain length, the thread
runner appends the policy to the queue.
"""
    last_state = model_name_to_process[brain](env.reset())

    last_features = policy.get_initial_features()
    length = 0
    rewards = 0
    openai_rewards = 0

    pc = None
    multiplier = None
    pc_repeat_time = None
    pc_mult = None
    pc_thre = None
    pc_max_repeat_time = None
    if a3cp:
        pc = PC()
        multiplier = 1
        pc_repeat_time = 0
        pc_mult = 2.5
        pc_thre = 0.01
        pc_max_repeat_time = 1000

    while True:
        terminal_end = False
        rollout = PartialRollout()

        for i in range(num_local_steps):
            if brain not in one_input_brain:
                if i == 0:
                    last_4_frames = [last_state[:, :, 0], last_state[:, :, 0], last_state[:, :, 0], last_state[:, :, 0]]
                else:
                    last_4_frames = [last_state[:, :, 0]] + last_4_frames[:3]

                fetched = policy.act(last_4_frames)
                action, value_ = fetched[0], fetched[1]
            else:
                fetched = policy.act(last_state, *last_features)
                action, value_, features = fetched[0], fetched[1], fetched[2:]

            # argmax to convert from one-hot
            state, openai_reward, terminal, info = env.step(action.argmax())
            if a3cp:
                pc_reward = pc.pc_reward(state) * multiplier
                reward = pc_reward + openai_reward
                if pc_mult:
                    if pc_reward < pc_thre:
                        pc_repeat_time += 1
                    else:
                        pc_repeat_time = 0
                    if pc_repeat_time >= pc_max_repeat_time:
                        multiplier *= pc_mult
                        pc_repeat_time = 0
                        print('Multiplier for pc reward is getting bigger. Multiplier=' + str(multiplier))
            else:
                reward = openai_reward

            state = model_name_to_process[brain](state)
            if render:
                env.render()

            # collect the experience
            if brain not in one_input_brain:
                rollout.add(last_4_frames, action, reward, value_, terminal, last_features)
            else:
                rollout.add(last_state, action, reward, value_, terminal, last_features)
            length += 1
            rewards += reward
            openai_rewards += openai_reward

            last_state = state
            if brain in one_input_brain:
                last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if terminal or length >= timestep_limit:
                terminal_end = True
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
                    last_state = model_name_to_process[brain](last_state)
                last_features = policy.get_initial_features()
                if a3cp:
                    print("Episode finished. Sum of game rewards: %d. PC-reward: %d Length: %d" % (
                        openai_rewards, rewards, length))
                else:
                    print("Episode finished. Sum of rewards: %d. Length: %d" % (openai_rewards, length))

                summary = tf.Summary()
                summary.value.add(tag="episode/reward", simple_value=float(openai_rewards))
                if a3cp:
                    summary.value.add(tag="episode/PC-reward", simple_value=float(rewards))
                summary.value.add(tag="episode/length", simple_value=float(length))
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()

                length = 0
                rewards = 0
                openai_rewards = 0
                break

        if not terminal_end:
            if brain not in one_input_brain:
                rollout.r = policy.value(last_4_frames)
            else:
                rollout.r = policy.value(last_state, *last_features)

        # once we have enough experience, yield it, and have the ThreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(self, env, task, visualise, visualiseVIN, brain, final_learning_rate, local_steps, a3cp, initial_lr=0,
                 max_t=0):
        """
An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
should be computed.
"""
        self.initial_lr = initial_lr
        self.final_lr = final_learning_rate
        self.max_t = max_t

        self.brain = brain
        self.env = env
        self.task = task
        self.visualiseVIN = visualiseVIN
        worker_device = "/job:worker/task:{}/cpu:0".format(task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                if brain in possible_model:
                    self.network = model_name_to_class[brain](env.observation_space.shape, env.action_space.n)
                else:
                    print("Unknown brain structure")
                self.global_step = tf.get_variable("global_step", [], tf.int32,
                                                   initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                   trainable=False)

        with tf.device(worker_device):
            with tf.variable_scope("local"):
                if brain in possible_model:
                    self.local_network = pi = model_name_to_class[brain](env.observation_space.shape,
                                                                         env.action_space.n)
                else:
                    print("Unknown brain structure")

                pi.global_step = self.global_step

            self.ac = tf.placeholder(tf.float32, [None, env.action_space.n], name="ac")
            self.adv = tf.placeholder(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")

            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)

            # the "policy gradients" loss:  its derivative is precisely the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = - tf.reduce_sum(tf.reduce_sum(log_prob_tf * self.ac, [1]) * self.adv)

            # loss of value function
            vf_loss = 0.5 * tf.reduce_sum(tf.square(pi.vf - self.r))
            entropy = - tf.reduce_sum(prob_tf * log_prob_tf)

            bs = tf.to_float(tf.shape(pi.x)[0])
            self.loss = pi_loss + 0.5 * vf_loss - entropy * 0.01

            # 20 represents the number of "local steps":  the number of timesteps
            # we run the policy before we update the parameters.
            # The larger local steps is, the lower is the variance in our policy gradients estimate
            # on the one hand;  but on the other hand, we get less frequent parameter updates, which
            # slows down learning.  In this code, we found that making local steps be much
            # smaller than 20 makes the algorithm more difficult to tune and to get to work.
            self.runner = RunnerThread(env, pi, local_steps, visualise, brain, a3cp)

            grads = tf.gradients(self.loss, pi.var_list)
            grads, _ = tf.clip_by_global_norm(grads, 40.0)

            # copy weights from the parameter server to the local model
            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

            grads_and_vars = list(zip(grads, self.network.var_list))
            inc_step = self.global_step.assign_add(tf.shape(pi.x)[0])

            # each worker has a different set of adam optimizer parameters
            self.lr = tf.placeholder(tf.float32)
            # opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.99)
            # opt = tf.train.RMSPropOptimizer(self.lr, decay=0.99, momentum=0.0, epsilon=0.1, use_locking=False)
            opt = tf.train.AdamOptimizer(self.lr)

            self.train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
            self.summary_writer = None
            self.local_steps = 0

            tf.summary.scalar("model/policy_loss", pi_loss / bs)
            tf.summary.scalar("model/value_loss", vf_loss / bs)
            tf.summary.scalar("model/entropy", entropy / bs)
            if brain in one_input_brain:
                tf.summary.image("model/state", pi.x)
            tf.summary.scalar("model/grad_global_norm", tf.global_norm(grads))
            tf.summary.scalar("model/var_global_norm", tf.global_norm(pi.var_list))
            tf.summary.scalar("model/lr", self.lr)
            self.summary_op = tf.summary.merge_all()

    def start(self, sess, summary_writer):
        self.runner.start_runner(sess, summary_writer)
        self.summary_writer = summary_writer

    def pull_batch_from_queue(self):
        """
self explanatory:  take a rollout from the queue of the thread runner.
"""
        rollout = self.runner.queue.get(timeout=600.0)
        while not rollout.terminal:
            try:
                rollout.extend(self.runner.queue.get_nowait())
            except queue.Empty:
                break
        return rollout

    def process(self, sess):
        """
process grabs a rollout that's been produced by the thread runner,
and updates the parameters.  The update is then sent to the parameter
server.
"""

        sess.run(self.sync)  # copy weights from shared to local
        rollout = self.pull_batch_from_queue()
        batch = process_rollout(rollout, gamma=0.99, lambda_=1.0)

        should_compute_summary = self.task == 0 and self.local_steps % 100 == 0

        if should_compute_summary:
            fetches = [self.summary_op, self.train_op, self.global_step]
        else:
            fetches = [self.train_op, self.global_step]

        # Visualise self.local_network.r and self.local_network.state
        if self.visualiseVIN:
            fetches += [self.local_network.reward, self.local_network.state]

        cur_global_step = self.global_step.eval()
        if self.brain not in one_input_brain:
            feed_dict = {
                self.lr: current_lr(cur_global_step, self.max_t, self.initial_lr, self.final_lr),
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
            }
        else:
            feed_dict = {
                self.lr: current_lr(cur_global_step, self.max_t, self.initial_lr, self.final_lr),
                self.local_network.x: batch.si,
                self.ac: batch.a,
                self.adv: batch.adv,
                self.r: batch.r,
                self.local_network.state_in[0]: batch.features[0],
                self.local_network.state_in[1]: batch.features[1],
            }

        fetched = sess.run(fetches, feed_dict=feed_dict)

        # Visualise self.local_network.r and self.local_network.state
        if self.visualiseVIN:
            print("r:", fetched[-2][0])
            print("state:", fetched[-1][0])
            X = np.linspace(0, 160, 160, endpoint=False)
            plt.subplot(211)
            # Normalize data
            reward_plot = fetched[-2][0] / np.max(fetched[-2][0])
            state_plot = fetched[-1][0] / np.max(fetched[-1][0])
            plt.plot(X, reward_plot, color="blue", linestyle="-", label="Reward")
            plt.plot(X, state_plot, color="red", linestyle="-", label="State")
            plt.legend(loc='upper left')
            plt.subplot(212)
            if self.brain not in one_input_brain:
                plt.imshow(batch.si[0, 0, :, :])
            else:
                plt.imshow(batch.si[0, :, :, 0])
            plt.show()

        if should_compute_summary:
            self.summary_writer.add_summary(tf.Summary.FromString(fetched[0]), fetched[2])
            self.summary_writer.flush()
        self.local_steps += 1
