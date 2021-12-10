import tensorflow as tf
import tf_util_extend as TFUtilExtend

NAME = "fc_3layers_512units_256units_reduced_state_gating"

def build_net(input_tfs, reuse=False):
    state_layers = [128, 32]
    goal_layers  = [512, 256]
    combined_layer = 256
    activation = tf.nn.relu
    net_count = 0

    # State branch
    h_state, net_count =  TFUtilExtend.fc_net(input_tfs[0], net_count, state_layers, activation=activation, reuse=reuse)
    h_state = activation(h_state)

    # Goal branch
    h_goal, net_count = TFUtilExtend.fc_net(input_tfs[1], net_count, goal_layers, activation=activation, reuse=reuse)
    h_goal = activation(h_goal)

    # Combine the the input branches
    h = tf.concat(axis=-1, values=[h_state, h_goal])
    h, _ = TFUtilExtend.fc_net(h, net_count, [combined_layer], activation=activation, reuse=reuse)
    h = activation(h)

    return h