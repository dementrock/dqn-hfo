import os
# Hide INFO and WARNING, show ERROR and FATAL
os.environ['GLOG_minloglevel'] = '2'
from caffe_ext import layers as L
from caffe.proto.caffe_pb2 import SolverParameter
del os.environ['GLOG_minloglevel']

import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../build/libdqn-c-lib.so"
    )
)

lib.DQN_new.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_float,
]
lib.DQN_new.returntype = ctypes.c_void_p

lib.DQN_SelectAction.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.DQN_SelectAction.returntype = None

lib.DQN_AddTransition.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_float,
    ctypes.POINTER(ctypes.c_float),
]
lib.DQN_AddTransition.returntype = None

lib.DQN_Update.argtypes = [ctypes.c_void_p]
lib.DQN_Update.returntype = None

lib.DQN_actor_net.argtypes = [ctypes.c_void_p]
lib.DQN_actor_net.returntype = ctypes.c_void_p

lib.DQN_critic_net.argtypes = [ctypes.c_void_p]
lib.DQN_critic_net.returntype = ctypes.c_void_p

lib.DQN_actor_target_net.argtypes = [ctypes.c_void_p]
lib.DQN_actor_target_net.returntype = ctypes.c_void_p

lib.DQN_critic_target_net.argtypes = [ctypes.c_void_p]
lib.DQN_critic_target_net.returntype = ctypes.c_void_p

lib.Net_print_diagnostics.argtypes = [ctypes.c_void_p]
lib.Net_print_diagnostics.returntype = None


class Net(object):

    def __init__(self, obj):
        self.obj = obj

    def print_diagnostics(self):
        lib.Net_print_diagnostics(self.obj)


class DQN(object):
    def __init__(
            self,
            state_size,
            action_size,
            save_path="/dev/null",
            minibatch_size=32,
            replay_pool_size=1000000,
            memory_threshold=10000,
            discount=0.99,
            actor_h1_size=400,
            actor_h2_size=300,
            critic_h1_size=400,
            critic_h2_size=300,
            actor_learning_rate=1e-5,
            critic_learning_rate=1e-3):
        actor_solver_param = new_solver(learning_rate=actor_learning_rate)
        actor_net_param = new_actor_net_param(
            state_size=state_size,
            action_size=action_size,
            minibatch_size=minibatch_size,
            h1_size=actor_h1_size,
            h2_size=actor_h2_size
        )
        critic_solver_param = new_solver(learning_rate=critic_learning_rate)
        critic_solver_param.weight_decay = 0.01
        critic_net_param = new_critic_net_param(
            state_size=state_size,
            action_size=action_size,
            minibatch_size=minibatch_size,
            h1_size=critic_h1_size,
            h2_size=critic_h2_size,
        )
        obj = lib.DQN_new(
            actor_solver_param.SerializeToString(),
            len(actor_solver_param.SerializeToString()),
            actor_net_param.SerializeToString(),
            len(actor_net_param.SerializeToString()),
            critic_solver_param.SerializeToString(),
            len(critic_solver_param.SerializeToString()),
            critic_net_param.SerializeToString(),
            len(critic_net_param.SerializeToString()),
            save_path,
            state_size,
            action_size,
            minibatch_size,
            replay_pool_size,
            memory_threshold,
            discount
        )
        self.obj = obj
        self.state_size = state_size
        self.action_size = action_size
        self.minibatch_size = minibatch_size
        self.replay_pool_size = replay_pool_size
        self.discount = discount

    def select_action(self, state):
        state = np.array(state, dtype=np.float32)
        action = np.zeros(self.action_size, dtype=np.float32)
        lib.DQN_SelectAction(
            self.obj,
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            action.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return action

    def update(self):
        lib.DQN_Update(self.obj)

    def add_transition(self, state, action, reward, next_state):
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        reward = np.float32(reward)
        if next_state is not None:
            next_state = np.array(next_state, dtype=np.float32)
            next_state_pt = next_state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            next_state_pt = None
        lib.DQN_AddTransition(
            self.obj,
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            action.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            reward,
            next_state_pt
        )

    @property
    def actor_net(self):
        return Net(lib.DQN_actor_net(self.obj))

    @property
    def critic_net(self):
        return Net(lib.DQN_critic_net(self.obj))

    @property
    def actor_target_net(self):
        return Net(lib.DQN_actor_target_net(self.obj))

    @property
    def critic_target_net(self):
        return Net(lib.DQN_critic_target_net(self.obj))

def new_solver(learning_rate):
    param = SolverParameter()
    param.solver_type = SolverParameter.ADAM
    param.momentum = 0.95
    param.base_lr = learning_rate
    param.lr_policy = "step"
    param.gamma = 0.1
    param.stepsize = 10000000
    param.max_iter = 10000000
    param.display = 0
    param.clip_gradients = 10
    return param


def new_actor_net_param(state_size, action_size, minibatch_size, h1_size, h2_size):
    state_input, _ = L.MemoryData(
        name="state_input_layer", 
        ntop=2,
        top_names=["states", "dummy1"],
        batch_size=minibatch_size,
        channels=1,
        height=state_size,
        width=1,
    )
    #state_input = L.BatchNorm(state_input)
    a1 = L.InnerProduct(state_input, num_output=h1_size, weight_filler=dict(type="xavier"))
    #a1 = L.BatchNorm(a1)
    h1 = L.ReLU(a1)#, negative_slope=0.01)
    a2 = L.InnerProduct(h1, num_output=h2_size, weight_filler=dict(type="xavier"))
    #a2 = L.BatchNorm(a2)
    h2 = L.ReLU(a2)#, negative_slope=0.01)
    a3 = L.InnerProduct(h2, num_output=action_size, weight_filler=dict(type="uniform", min=-3e-3, max=3e-3))
    #a3 = L.BatchNorm(a3)
    output = L.TanH(
        a3,
        name="actionpara_layer",
        top_name="action_params"
    )
    proto = output.to_proto()
    proto.force_backward = True
    # proto.layer.extend(dummy_layer.to_proto().layer)
    return proto


def new_critic_net_param(state_size, action_size, minibatch_size, h1_size, h2_size):
    state_input_layer, _ = L.MemoryData(
        name="state_input_layer",
        ntop=2,
        top_names=["states", "dummy1"],
        batch_size=minibatch_size,
        channels=1,
        height=state_size,
        width=1,
    )
    #state_input_layer = L.BatchNorm(state_input_layer)
    action_params_input_layer, _ = L.MemoryData(
        name="action_params_input_layer",
        ntop=2,
        top_names=["action_params", "dummy3"],
        batch_size=minibatch_size,
        channels=1,
        height=action_size,
        width=1,
    )
    #action_params_input_layer = L.BatchNorm(action_params_input_layer)
    action_params_input_layer = L.Reshape(
        action_params_input_layer,
        reshape_param=dict(shape=dict(dim=[minibatch_size, action_size])),
    )

    target_input_layer, _ = L.MemoryData(
        name="target_input_layer",
        ntop=2,
        top_names=["target", "dummy4"],
        batch_size=minibatch_size,
        channels=1,
        height=action_size,
        width=1,
    )

    a1 = L.InnerProduct(state_input_layer, num_output=h1_size, weight_filler=dict(type="xavier"))
    #a1 = L.BatchNorm(a1)
    h1  = L.ReLU(a1)#, negative_slope=0.01)

    inputs = L.Concat(
        h1, action_params_input_layer,
        axis=1
    )

    a2 = L.InnerProduct(inputs, num_output=h2_size, weight_filler=dict(type="xavier"))
    #a2 = L.BatchNorm(a2)
    h2 = L.ReLU(a2)#, negative_slope=0.01)
    q_values_layer = L.InnerProduct(
        h2,
        name="q_values_layer",
        top_name="q_values",
        num_output=1,
        weight_filler=dict(type="uniform", min=-3e-3, max=3e-3)
    )

    loss_layer = L.EuclideanLoss(
        q_values_layer, target_input_layer,
        name="loss",
        top_name="loss"
    )

    proto = loss_layer.to_proto()
    proto.force_backward = True
    return proto
