import ctypes
from caffe_ext import layers as L
from caffe.proto.caffe_pb2 import SolverParameter
import numpy as np

param = SolverParameter()
param.solver_type = param.ADAM

lib = ctypes.cdll.LoadLibrary("/root/code/dqn-info/build/libdqn-c-lib.so")
lib.DQN_new.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
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



class DQN(object):
    def __init__(
            self,
            actor_solver_param,
            actor_net_param,
            critic_solver_param,
            critic_net_param,
            save_path,
            state_size,
            action_size,
            minibatch_size,
            replay_pool_size,
            memory_threshold,
            discount):
        obj = lib.DQN_new(
            actor_solver_param.SerializeToString(),
            actor_net_param.SerializeToString(),
            critic_solver_param.SerializeToString(),
            critic_net_param.SerializeToString(),
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

def new_actor_solver():
    return SolverParameter(
        solver_type=SolverParameter.ADAM,
        momentum=0.95,
        base_lr=0.00001,
        lr_policy="step",
        gamma=0.1,
        stepsize=10000000,
        max_iter=10000000,
        display=0,
        clip_gradients=10
    )


def new_actor_net_param(obs_dim, action_dim, minibatch_size):
    state_input = L.MemoryData(
        name="state_input_layer", 
        ntop=2,
        top_names=["states", "dummy1"],
        batch_size=minibatch_size,
        channels=1,
        height=obs_dim,
        width=1,
    )
    a1 = L.InnerProduct(state_input[0], num_output=400, weight_filler=dict(type="gaussian", std=0.01))
    h1 = L.ReLU(a1, negative_slope=0.01)
    a2 = L.InnerProduct(h1, num_output=300, weight_filler=dict(type="gaussian", std=0.01))
    h2 = L.ReLU(a2, negative_slope=0.01)
    a3 = L.InnerProduct(h2, num_output=action_dim, weight_filler=dict(type="gaussian", std=0.01))
    output = L.TanH(
        a3,
        name="actionpara_layer",
        top_name="action_params"
    )
    proto = output.to_proto()
    # proto.layer.extend(dummy_layer.to_proto().layer)
    return proto


def new_critic_net_param(obs_dim, action_dim, minibatch_size):
    state_input_layer, _ = L.MemoryData(
        name="state_input_layer",
        ntop=2,
        top_names=["states", "dummy1"],
        batch_size=minibatch_size,
        channels=1,
        height=obs_dim,
        width=1,
    )
    action_params_input_layer, _ = L.MemoryData(
        name="action_params_input_layer",
        ntop=2,
        top_names=["action_params", "dummy3"],
        batch_size=minibatch_size,
        channels=1,
        height=action_dim,
        width=1,
    )

    target_input_layer, _ = L.MemoryData(
        name="target_input_layer",
        ntop=2,
        top_names=["target", "dummy4"],
        batch_size=minibatch_size,
        channels=1,
        height=action_dim,
        width=1,
    )

    inputs = L.Concat(
        state_input_layer, action_params_input_layer,
        axis=2
    )
    # dummy_layer_1 = L.Silence(state_input_layer)
    # dummy_layer_2 = L.Silence(action_input_layer)
    a1 = L.InnerProduct(inputs, num_output=400, weight_filler=dict(type="gaussian", std=0.01))
    h1  = L.ReLU(a1, negative_slope=0.01)
    a2 = L.InnerProduct(h1, num_output=300, weight_filler=dict(type="gaussian", std=0.01))
    h2 = L.ReLU(a2, negative_slope=0.01)
    q_values_layer = L.InnerProduct(
        h2,
        name="q_values_layer",
        top_name="q_values",
        num_output=1,
        weight_filler=dict(type="gaussian", std=0.01)
    )

    loss_layer = L.EuclideanLoss(
        q_values_layer, target_input_layer,
        name="loss",
        top_name="loss"
    )

    proto = loss_layer.to_proto()
    # proto.layer.extend(dummy_layer_1.to_proto().layer)
    # proto.layer.extend(dummy_layer_2.to_proto().layer)
    return proto


def new_critic_solver():
    return SolverParameter(
        solver_type=SolverParameter.ADAM,
        momentum=0.95,
        base_lr=0.001,
        lr_policy="step",
        gamma=0.1,
        stepsize=10000000,
        max_iter=10000000,
        display=0,
        clip_gradients=10,
    )

obs_dim = 4
action_dim = 1
minibatch_size = 32
replay_pool_size = 1000000
min_pool_size = 10000
discount = 0.99

actor_solver_param = new_actor_solver()
actor_net_param = new_actor_net_param(obs_dim, action_dim, minibatch_size)
critic_solver_param = new_critic_solver()
critic_net_param = new_critic_net_param(obs_dim, action_dim, minibatch_size)

dqn = DQN(
    actor_solver_param,
    actor_net_param,
    critic_solver_param,
    critic_net_param,
    save_path="log",
    state_size=obs_dim,
    action_size=action_dim,
    minibatch_size=minibatch_size,
    replay_pool_size=replay_pool_size,
    memory_threshold=min_pool_size,
    discount=discount,
)
print dqn.select_action([1,2,3,4])
dqn.add_transition([1,2,3,4], [1], 1, [1,2,3,5])
dqn.update()
import ipdb; ipdb.set_trace()
