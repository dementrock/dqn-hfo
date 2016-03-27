#include "dqn.hpp"
#include "dqn.hpp"
#include <caffe/caffe.hpp>
#include "stdio.h"

extern "C" {
  //void print_solver_type(char* param_str) {
  //    caffe::SolverParameter param;
  //    param.ParseFromString(param_str);
  //    //caffe::SolverParameter::ParseFromString(param_str);
  //    printf("%d\n", param.solver_type());
  //  //printf("%d\n", solver_param->solver_type());
  //}
  dqn::DQN *DQN_new(
          char* actor_solver_param_str,
          int actor_solver_param_strlen,
          char* actor_net_param_str,
          int actor_net_param_strlen,
          char* critic_solver_param_str,
          int critic_solver_param_strlen,
          char* critic_net_param_str,
          int critic_net_param_strlen,
          char* save_path,
          int state_size,
          int action_param_size,
          int minibatch_size,
          int replay_memory_capacity,
          int memory_threshold,
          float gamma) {
    std::string actor_solver_param_s(actor_solver_param_str, actor_solver_param_strlen);
    std::string actor_net_param_s(actor_net_param_str, actor_net_param_strlen);
    std::string critic_solver_param_s(critic_solver_param_str, critic_solver_param_strlen);
    std::string critic_net_param_s(critic_net_param_str, critic_net_param_strlen);
    caffe::SolverParameter actor_solver_param;
    actor_solver_param.ParseFromString(actor_solver_param_s);
    //std::cout << "here!" << std::endl;
    //std::cout << actor_solver_param_s << std::endl;
    //std::cout << actor_solver_param.solver_type() << std::endl;

    actor_solver_param.mutable_net_param()->ParseFromString(actor_net_param_s);
    caffe::SolverParameter critic_solver_param;
    critic_solver_param.ParseFromString(critic_solver_param_s);
    critic_solver_param.mutable_net_param()->ParseFromString(critic_net_param_s);
    //std::cout << actor_solver_param.lr_policy() << std::endl;
    std::string save_path_str(save_path);
    return new dqn::DQN(
        actor_solver_param,
        critic_solver_param,
        save_path_str,
        state_size,
        action_param_size,
        minibatch_size,
        replay_memory_capacity,
        memory_threshold,
        gamma
    );
  }

  void DQN_Update(dqn::DQN* dqn) {
      dqn->Update();
  }

  void DQN_AddTransition(dqn::DQN* dqn, float* state, float* action, float reward, float* next_state) {
    dqn::StateDataSp state_data = std::make_shared<dqn::StateData>(dqn->state_size());
    dqn::ActorOutput action_data(dqn->action_param_size());
    std::copy(state, state + dqn->state_size(), state_data->begin());
    std::copy(action, action + dqn->action_param_size(), action_data.begin());
    if (next_state != NULL) {
        dqn::StateDataSp next_state_data = std::make_shared<dqn::StateData>(dqn->state_size());
        std::copy(next_state, next_state + dqn->state_size(), next_state_data->begin());
        dqn->AddTransition(
            dqn::Transition(state_data, action_data, reward, next_state_data)
        );
    } else {
        dqn->AddTransition(
            dqn::Transition(state_data, action_data, reward, boost::none)
        );
    }
  }

  void DQN_SelectAction(dqn::DQN* dqn, float* state, float* action_output) {
      dqn::StateDataSp state_data = std::make_shared<dqn::StateData>(dqn->state_size());
      std::copy(state, state+dqn->state_size(), state_data->begin());
      //for (int i = 0; i < state_data->size(); ++i) {
      //    printf("%f\n", (*state_data)[i]);
      //}
      const dqn::ActorOutput& output = dqn->SelectAction(state_data, 0.);
      //printf("got output! copying...\n");
      std::copy(output.begin(), output.end(), action_output);
      //printf("copied!\n");
  }

  caffe::Net<float>* DQN_actor_net(dqn::DQN* dqn) {
    return &(*dqn->actor_net());
  }

  caffe::Net<float>* DQN_critic_net(dqn::DQN* dqn) {
    return &(*dqn->critic_net());
  }

  caffe::Net<float>* DQN_actor_target_net(dqn::DQN* dqn) {
    return &(*dqn->actor_target_net());
  }

  caffe::Net<float>* DQN_critic_target_net(dqn::DQN* dqn) {
    return &(*dqn->critic_target_net());
  }

  void Net_print_diagnostics(caffe::Net<float>* net) {
    const auto& params = net->learnable_params();
    float sqsum = 0;
    for (int i = 0; i < params.size(); ++i) {
      int count = params[i]->count();
      for (int j = 0; j < count; ++j) {
        float val = params[i]->cpu_data()[j];
        sqsum += val * val;
      }
    }
    std::cout << "sqsum: " << sqsum << std::endl;
  }

}
