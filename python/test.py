from dqn import DQN

state_size = 4
action_size = 1
#minibatch_size = 32
#replay_pool_size = 1000000
#min_pool_size = 10000
#discount = 0.99
#actor_learning_rate = 1e-5
#critic_learning_rate = 1e-3

dqn = DQN(
    #actor_solver_param,
    #actor_net_param,
    #critic_solver_param,
    #critic_net_param,
    state_size=state_size,
    action_size=action_size,
    #minibatch_size=minibatch_size,
    #replay_pool_size=replay_pool_size,
    #memory_threshold=min_pool_size,
    #discount=discount,
)
print dqn.select_action([1,2,3,4])
dqn.add_transition([1,2,3,4], [1], 1, [1,2,3,5])
dqn.update()
