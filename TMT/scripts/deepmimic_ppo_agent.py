import numpy as np
import copy as copy
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from env.env import Env
from learning.ppo_agent import PPOAgent
from learning.solvers.mpi_solver import MPISolver
import learning.tf_util as TFUtil
import tf_util_extend as TFUtilExtend
from learning.tf_normalizer import TFNormalizer
import learning.rl_util as RLUtil
from util.logger import Logger
import util.mpi_util as MPIUtil
import mpi_util_extend as MPIUtilExtend
import util.math_util as MathUtil
from env.env import Env
import net_builder as NetBuilder


class DeepMimicPPOAgent(PPOAgent):
    NAME = "DeepMimicPPO"
    MOTION_NAMES_KEY = 'MotionNames'

    def __init__(self, world, id, json_data, seed):
        self.motion_names = []
        self.s_norms = []
        self.g_norms = []
        self.a_norms = []
        self.a_mean_tfs = []
        self.random_seed = seed
        super().__init__(world, id, json_data)
        return

    def _load_params(self, json_data):
        super()._load_params(json_data)

        if self.MOTION_NAMES_KEY in json_data:
            self.motion_names = json_data[self.MOTION_NAMES_KEY]
        return

    def _build_nets(self, json_data):
        assert self.ACTOR_NET_KEY in json_data
        assert self.CRITIC_NET_KEY in json_data

        actor_net_name = json_data[self.ACTOR_NET_KEY]
        critic_net_name = json_data[self.CRITIC_NET_KEY]
        actor_init_output_scale = 1 if (self.ACTOR_INIT_OUTPUT_SCALE_KEY not in json_data) else json_data[self.ACTOR_INIT_OUTPUT_SCALE_KEY]

        s_size = self.get_state_size()
        g_size = self.get_goal_size()
        a_size = self.get_action_size()

        # setup input tensors
        self.s_tf = tf.placeholder(tf.float32, shape=[None, s_size], name="s")
        self.a_tf = tf.placeholder(tf.float32, shape=[None, a_size], name="a")
        self.tar_val_tf = tf.placeholder(tf.float32, shape=[None], name="tar_val")
        self.adv_tf = tf.placeholder(tf.float32, shape=[None], name="adv")
        self.g_tf = tf.placeholder(tf.float32, shape=([None, g_size] if self.has_goal() else None), name="g")
        self.old_logp_tf = tf.placeholder(tf.float32, shape=[None], name="old_logp")
        self.exp_mask_tf = tf.placeholder(tf.float32, shape=[None], name="exp_mask")

        with tf.variable_scope('main'):
            # build actor net
            with tf.variable_scope('actor'):
                if len(self.motion_names) > 0:
                    self.a_mean_tfs = [None] * len(self.motion_names)
                    for i, motion_name in enumerate(self.motion_names):
                        self.a_mean_tfs[i] = self._build_net_actor(motion_name, actor_net_name, actor_init_output_scale)
                    self.a_mean_tf = self.a_mean_tfs[0]
                else:
                    self.a_mean_tf = self._build_net_actor(actor_net_name, actor_init_output_scale)

            # built critic net
            with tf.variable_scope('critic'):
                self.critic_tf = self._build_net_critic(critic_net_name)

        if (self.a_mean_tf != None):
            Logger.print('Built actor net: ' + actor_net_name)

        if (self.critic_tf != None):
            Logger.print('Built critic net: ' + critic_net_name)

        self.norm_a_std_tf = self.exp_params_curr.noise * tf.ones(a_size)
        norm_a_noise_tf = self.norm_a_std_tf * tf.random_normal(shape=tf.shape(self.a_mean_tf))
        norm_a_noise_tf *= tf.expand_dims(self.exp_mask_tf, axis=-1)
        self.sample_a_tf = self.a_mean_tf + norm_a_noise_tf * self.a_norm.std_tf
        self.sample_a_logp_tf = TFUtil.calc_logp_gaussian(x_tf=norm_a_noise_tf, mean_tf=None, std_tf=self.norm_a_std_tf)

        self.sample_a_tfs = [None] * len(self.motion_names)
        self.sample_a_logp_tfs = [None] * len(self.motion_names)
        for i in range(0, len(self.motion_names)):
            self.sample_a_tfs[i] = self.a_mean_tfs[i] + norm_a_noise_tf * self.a_norm.std_tf
            self.sample_a_logp_tfs[i] = self.sample_a_logp_tf
        return

    def _build_net_actor(self, motion_name, net_name, init_output_scale, reuse=False):
        with tf.variable_scope(motion_name, reuse=reuse):
            motion_idx = self.motion_names.index(motion_name)
            input_tfs = self._get_actor_inputs(motion_idx)

            h = NetBuilder.build_net(net_name, input_tfs)
            norm_a_tf = tf.layers.dense(inputs=h,
                                        units=self.get_action_size(),
                                        activation=None,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-init_output_scale, maxval=init_output_scale))
            a_tf = self.a_norms[motion_idx].unnormalize_tf(norm_a_tf)
        return a_tf

    def _get_actor_inputs(self, motion_idx=-1):
        if motion_idx >= 0 and motion_idx < len(self.s_norms):
            norm_s_tf = self.s_norms[motion_idx].normalize_tf(self.s_tf)
            input_tfs = [norm_s_tf]
            if self.has_goal():
                norm_g_tf = self.g_norms[motion_idx].normalize_tf(self.g_tf)
                input_tfs += [norm_g_tf]
        else:
            norm_s_tf = self.s_norm.normalize_tf(self.s_tf)
            input_tfs = [norm_s_tf]
            if self.has_goal():
                norm_g_tf = self.g_norm.normalize_tf(self.g_tf)
                input_tfs += [norm_g_tf]
        return input_tfs

    def _build_normalizers(self):
        self.s_norms = [None] * len(self.motion_names)
        self.g_norms = [None] * len(self.motion_names)
        self.a_norms = [None] * len(self.motion_names)
        with self.sess.as_default(), self.graph.as_default(), tf.variable_scope(self.tf_scope):
            with tf.variable_scope(self.RESOURCE_SCOPE):
                for i, name in enumerate(self.motion_names):
                    with tf.variable_scope(name):
                        s_norm = TFNormalizer(self.sess, 's_norm', self.get_state_size(), self.world.env.build_state_norm_groups(self.id))
                        s_norm.set_mean_std(-self.world.env.build_state_offset(self.id), 1 / self.world.env.build_state_scale(self.id))
                        self.s_norms[i] = s_norm

                        g_norm = TFNormalizer(self.sess, 'g_norm', self.get_goal_size(), self.world.env.build_goal_norm_groups(self.id))
                        g_norm.set_mean_std(-self.world.env.build_goal_offset(self.id), 1 / self.world.env.build_goal_scale(self.id))
                        self.g_norms[i] = g_norm

                        a_norm = TFNormalizer(self.sess, 'a_norm', self.get_action_size())
                        a_norm.set_mean_std(-self.world.env.build_action_offset(self.id), 1 / self.world.env.build_action_scale(self.id))
                        self.a_norms[i] = a_norm
        return super()._build_normalizers()

    def _init_normalizers(self):
        with self.sess.as_default(), self.graph.as_default():
            for i in range(0, len(self.motion_names)):
                self.s_norms[i].load()
                self.g_norms[i].load()
                self.a_norms[i].load()
        return super()._init_normalizers()

    def _load_normalizers(self):
        for i in range(0, len(self.motion_names)):
            self.s_norms[i].load()
            self.g_norms[i].load()
            self.a_norms[i].load()
        return super()._load_normalizers()

    def _update_normalizers(self):
        for i in range(0, len(self.motion_names)):
            self.s_norms[i].update()
            if self.has_goal():
                self.g_norms[i].update()
        return super()._update_normalizers()

    def _record_motion_label(self):
        motion_label = self.world.env.record_motion_label(self.id)
        return motion_label

    def _eval_actor(self, s, g, enable_exp):
        s = np.reshape(s, [-1, self.get_state_size()])
        g = np.reshape(g, [-1, self.get_goal_size()]) if self.has_goal() else None
        motion_label = self._record_motion_label()

        feed = {
            self.s_tf : s,
            self.g_tf : g,
            self.exp_mask_tf: np.array([1 if enable_exp else 0])
        }

        if len(self.motion_names) > 0:
            run_tfs = [self.sample_a_tfs[motion_label], self.sample_a_logp_tfs[motion_label]]
        else:
            run_tfs = [self.sample_a_tf, self.sample_a_logp_tf]

        a, logp = self.sess.run(run_tfs, feed_dict=feed)
        return a, logp

    def _update_new_action(self):
        s = self._record_state()
        g = self._record_goal()

        if self.enable_training and not (self._is_first_step()):
            r = self._record_reward()
            self.path.rewards.append(r)

        a, logp = self._decide_action(s=s, g=g)
        assert len(np.shape(a)) == 1
        assert len(np.shape(logp)) <= 1

        flags = self._record_flags()
        self._apply_action(a)

        if self.enable_training:
            self.path.states.append(s)
            self.path.goals.append(g)
            self.path.actions.append(a)
            self.path.logps.append(logp)
            self.path.flags.append(flags)

        if self._enable_draw():
            self._log_val(s, g)

        return

    def load_actor_models(self, motion_names, model_files):
        with self.sess.as_default(), self.graph.as_default():
            Logger.print("====================")
            Logger.print("Loading Actor Models")
            Logger.print("====================")

            for motion_name, model_file in zip(motion_names, model_files):
                if motion_name in motion_names:
                    Logger.print("Using %s -- %s" % (motion_name, model_file))
                    moiton_idx = self.motion_names.index(motion_name)

                    reader = pywrap_tensorflow.NewCheckpointReader(model_file)
                    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='agent/main/actor/{}'.format(motion_name))

                    # copy actor model parameters
                    for v in var_list:
                        lookfor = v.name.replace('/{}'.format(motion_name), '').replace(':0', '')
                        Logger.print("Assign {} --> {}".format(lookfor, v.name))
                        weights = reader.get_tensor(lookfor)
                        self.sess.run(v.assign(weights))

                    # copy normalizer parameters
                    s_mean, s_std = reader.get_tensor('agent/resource/s_norm/mean'), reader.get_tensor('agent/resource/s_norm/std')
                    g_mean, g_std = reader.get_tensor('agent/resource/g_norm/mean'), reader.get_tensor('agent/resource/g_norm/std')
                    a_mean, a_std = reader.get_tensor('agent/resource/a_norm/mean'), reader.get_tensor('agent/resource/a_norm/std')
                    self.s_norms[moiton_idx].set_mean_std(s_mean, s_std)
                    Logger.print("Set s_mean={}, s_std={}".format(np.mean(s_mean), np.mean(s_std)))
                    if self.has_goal():
                        self.g_norms[moiton_idx].set_mean_std(g_mean, g_std)
                        Logger.print("Set g_mean={}, g_std={}".format(np.mean(g_mean), np.mean(g_std)))
                    self.a_norms[moiton_idx].set_mean_std(a_mean, a_std)
                    Logger.print("Set a_mean={}, a_std={}".format(np.mean(a_mean), np.mean(a_std)))

                    self.s_norms[moiton_idx].update()
                    if self.has_goal():
                        self.g_norms[moiton_idx].update()
                    self.a_norms[moiton_idx].update()
            self._load_normalizers()

            for i, motion_name in enumerate(self.motion_names):
                Logger.print('----- Motion %s' % (motion_name))
                Logger.print(' State mean: %.10f, std: %.10f' % (np.mean(self.s_norms[i].mean), np.mean(self.s_norms[i].std)))
                if self.has_goal():
                    Logger.print('  Goal mean: %.10f, std: %.10f' % (np.mean(self.g_norms[i].mean), np.mean(self.g_norms[i].std)))
                Logger.print('Action mean: %.10f, std: %.10f' % (np.mean(self.a_norms[i].mean), np.mean(self.a_norms[i].std)))

            Logger.print("Sucessfully Load Actor Models")
        return
