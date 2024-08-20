<<<<<<< HEAD
from env.environnement import Environnement
from agents.Agent_BDQN import BDQN_Agent

if __name__=='__main__':
    #to specify: 
    nb_windturbines=6
    hiddens_common = [64, 32]
    hiddens_actions = [32, 16]
    hiddens_value = [64, 32]
    num_sub_actions = 3 #let it 3 at the moment
    batch_size=20
    N_epoch = 10
    #automatic generation
    env=Environnement.init(str(nb_windturbines)+"advanced")  # create environment 
    num_action_branches = nb_windturbines    
    aggregator = 'reduceLocalMean'    

    agent=BDQN_Agent(hiddens_common, hiddens_actions, hiddens_value, num_action_branches, num_sub_actions)
    agent.model.compile(loss='mse', optimizer='adam')
    agent.train(env,N_epoch,batch_size)
    print('training done')
    env.reset()
    agent_return=0
    print('initial state of environment:',env.state)
    for episode in range(50):
        action=agent.act(env.state)
        action_dict=env.convert_action_list_dict(action)
        _,rew= env.step(action_dict)
        agent_return+=rew
        print('return of agent:',agent_return )
=======
from env.environnement import Environnement
from agents.Agent_BDQN import BDQN_Agent

if __name__=='__main__':
    #to specify: 
    nb_windturbines=6
    hiddens_common = [64, 32]
    hiddens_actions = [32, 16]
    hiddens_value = [64, 32]
    num_sub_actions = 3 #let it 3 at the moment
    batch_size=20
    N_epoch = 10
    #automatic generation
    env=Environnement.init(str(nb_windturbines)+"advanced")  # create environment 
    num_action_branches = nb_windturbines    
    aggregator = 'reduceLocalMean'    

    agent=BDQN_Agent(hiddens_common, hiddens_actions, hiddens_value, num_action_branches, num_sub_actions)
    agent.model.compile(loss='mse', optimizer='adam')
    agent.train(env,N_epoch,batch_size)
    print('training done')
    env.reset()
    agent_return=0
    print('initial state of environment:',env.state)
    for episode in range(50):
        action=agent.act(env.state)
        action_dict=env.convert_action_list_dict(action)
        _,rew= env.step(action_dict)
        agent_return+=rew
        print('return of agent:',agent_return )

>>>>>>> ed4ca69 (Add files via upload)
        print('state of environment ep',episode,': ',env.state)
