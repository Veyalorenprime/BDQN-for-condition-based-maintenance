import tensorflow as tf
import tensorflow.keras.layers as layers
from collections import deque
from copy import deepcopy
import numpy as np
import random


class BDQN_Agent:
    """
    An implementation of the agent in Action Branching Architectures for Deep Reinforcement Learning
    
    The BDQN_Agent class provides methods for training the model on a given set of transitions, selecting actions in a given
    state, and updating the model's parameters during training.
    
    Attributes:
    - state_size (int): The number of features in the state representation.
    - num_sub_actions (int): The number of possible sub actions for each branch.
    - alpha (int )XXXXXXXXXXXXXXX  we should have it  XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX (float): The learning rate used by the optimizer during training.
    - gamma (float): The discount factor used to weight future rewards.
    - batch_size (int): The number of transitions sampled from the replay buffer for each training step.        XXXXXXXXXXX still not 100% clear
    - target_update_freq (int): The frequency (in steps) at which to update the target network.                 XXXXXXXXXXX is in the run, but is important to mention anywehere
    - tau (float): The interpolation factor used to update the target network parameters.                       XXXXXXXXXXX shouldn't it exist something like this in the fit??
    - epsilon (float): The current value of the epsilon-greedy exploration probability.
    - epsilon_min (float): The final value of the epsilon-greedy exploration probability.
    - epsilon_decay (float): The rate at which the epsilon-greedy exploration probability decays over time.
    - model (BayesianNetwork): The neural network used to approximate the Q-function.
    - target_network (BayesianNetwork): A copy of the network used for computing the target values.
    - optimizer (Optimizer): The optimizer used to update the network parameters.
    - replay_buffer (ReplayBuffer): The replay buffer used to store past transitions.
    - steps (int): The number of steps taken during training.                                                   XXXXXXXXXXXXX is like N_epochs
    """

    def __init__(self, hiddens_common, hiddens_actions, hiddens_value, num_action_branches, num_sub_actions):
        #network propreties
        self.state_size = num_action_branches # number of windturbines
        self.hiddens_common = hiddens_common
        self.hiddens_actions= hiddens_actions
        self.hiddens_value= hiddens_value
        self.num_action_branches= num_action_branches
        self.num_sub_actions= num_sub_actions
        self.aggregator='reduceLocalMean'
        #train propreties
        self.replay_buffer = deque(maxlen=1000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Build networks        
        
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.target_network = self.build_model()
        self.align_target_model()

    def build_model(self):
        """
        Builds and compiles the Deep Q-Network (DQN) model for the BDQN_Agent.

        Returns:
            A compiled keras model.
        """
        inpt=tf.keras.Input(shape=(self.num_action_branches,))
        out = inpt

        # Create the shared network module (unless independent)
        for hidden in self.hiddens_common:
            out = layers.Dense(units=hidden, activation=tf.nn.relu)(out)

        # Create the action branches
        total_action_scores = []
        for action_stream in range(self.num_action_branches):
            action_out = out
            for hidden in self.hiddens_actions:
                action_out = layers.Dense(units=hidden, activation=tf.nn.relu)(action_out)
            action_scores = layers.Dense(units=self.num_sub_actions, activation=None)(action_out)#change to num_sub_actions

            if self.aggregator == 'reduceLocalMean':
                action_scores_mean = tf.reduce_mean(action_scores, 1)
                total_action_scores.append(action_scores - tf.expand_dims(action_scores_mean, 1))
            else:
                total_action_scores.append(action_scores)    

        state_out = out
        for hidden in self.hiddens_value:
            state_out = layers.Dense(units=hidden, activation=tf.nn.relu)(state_out)
        state_score = layers.Dense( units=1, activation=None)(state_out)

        Q_values=[state_score + action_score for action_score in total_action_scores]

        model=tf.keras.Model(inputs=inpt, outputs=Q_values)
        return model
    
    def align_target_model(self):
        """
        Copies the weights from the main model to the target model.
        This function is called periodically to synchronize the target model with the main model.
        """
        self.target_network.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state):
        """
        Store a transition in the replay memory buffer.
        
        Args:
            state (numpy.ndarray): the current state of the environment.
            action (int): the action taken in the current state.
            reward (float): the reward received from taking the action.
            next_state (numpy.ndarray): the resulting state after taking the action.
            done (bool): whether the episode has terminated after taking the action.
        """
        # Store the transition in the replay memory buffer
        self.replay_buffer.append((state, action, reward, next_state))

    def act(self, state):
        """
        Given a state, selects an action according to the agent's policy.

        Args:
            state (np.ndarray): Current state of the environment.

        Returns:
            action (list): Action selected by the agent's policy.
        """
        if np.random.rand() <= self.epsilon:
            return state.getRandomAction()   # implement. also it can be: enviroment.action_space.sample()
            
        state=[item.wear for item in state.items]
        state = [np.asarray(state)]
        state = np.asarray(state)
        Q_values = self.model.predict(state)
        action=[]
        for q_item in Q_values:            
            lista = list(q_item[0])
            action.append(lista.index(max(lista)))
        return action

            
    def learn(self, batch_size):
        """
        Updates the main Q-network's weights using a batch of experiences sampled from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample from the replay buffer for each training iteration.

        Returns:
            None
        """ 
        if len(self.replay_buffer) < batch_size:
            return
        # Sample a batch of experiences from the replay buffer
        minibatch = random.sample(self.replay_buffer, batch_size)

        for state, action, reward, next_state in minibatch:
            
            #state = [state.items[0].wear, state.items[1].wear, state.items[2].wear]
            state=[item.wear for item in state.items]
            
            
            state = [np.asarray(state)]
            state = np.asarray(state)
            target = self.model.predict(state)
            
            for d in range(self.state_size):
                best_action = np.argmax(self.model.predict(next_state)[d])                  
                value_Q_target_d = self.target_network.predict(next_state)[d][0][best_action]
                target[d][0][action] = reward + (self.gamma/self.state_size) * value_Q_target_d
                
            self.model.fit(state, target, epochs=1, verbose=0)
            
        # Update the exploration-exploitation trade-off parameter (epsilon) 
        # and the learning rate (alpha). (XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX should do this last thing)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self,env,N_epoch,batch_size):
        """
        Trains the agent on the given environment. For every epoch, builds a replay buffer
          and learns from it.

        Args:
            env (Environnement): The environment to train the agent on.
            N_epoch (int): The number of epochs to train the agent for.
            batch_size (int): The size of the minibatch used for training the neural network.

            XXXXXXXXXXXXXXXXXXXXXXXX other thinks we might consider...
            discount_factor (float): The discount factor used for computing the Q-value targets.
            update_target_freq (int): The frequency (in number of episodes) with which to update the target network.

        Returns: XXXXXXXXXXXXXXXXXXXXXXXXXXX probably it should return something to give the final result
            rewards_history (list): A list containing the total rewards obtained for each episode during training.
        """   
        for i in range(N_epoch):
            state=env.get_random_state()
            batch = 0 

            self.batch_reset()  	           
            
            for batch in range(batch_size*2):# *2 to have more data in the replay buffer than needed
                print(i,'|batch:', batch)

                # Agent takes action
                action = self.act(state)
                action_dict = env.convert_action_list_dict(action)

                # Action has effect on environment
                current_state = deepcopy(state)

                next_state, reward = env.step(action_dict)

                #next_state_proper = [next_state.items[0].wear, next_state.items[1].wear, next_state.items[2].wear]
                next_state_proper=[item.wear for item in next_state.items]
                next_state_proper = [np.asarray(next_state_proper)]
                next_state_proper = np.asarray(next_state_proper)

                # Agent observe the transition and possibly learns
                self.remember(current_state, action, reward, next_state_proper)
              
                state = deepcopy(next_state)

            self.learn(batch_size)
            self.align_target_model()

    def batch_reset(self):
        """
        Resets the replay buffer and target network.
        """
        self.replay_buffer = deque(maxlen=500)
