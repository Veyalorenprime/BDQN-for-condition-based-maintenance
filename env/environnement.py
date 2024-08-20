from os import stat
from typing import List, Dict, Union, Tuple, Callable
import random
import numpy as np
from copy import deepcopy

#from project.env.actions import Action, CoreAction
from env.states import State
from env.items import Item
from env.executions import execution
from env.utils_action import IsDoNothing,count_prev_corr,IsAcceptable


class Environnement:
    def __init__(
        self,
        items: List[Item],
        prev_efficiency: float=2,
        repair_thrs: float = 0,
        ship_cost: float = 18,
        corr_cost: float = 18,
        prev_cost: float = 5,
    ) -> None:
        self.prev_efficiency = prev_efficiency
        self.repair_thrs = repair_thrs
        self.ship_cost = ship_cost
        self.corr_cost = corr_cost
        self.prev_cost = prev_cost
        
        self.line1 = []
        self.fig = None
        self.items = items
        self.state = State(items)
    
        self._initial_wears = []

        for item in self.items:
            self._initial_wears.append(item.wear)


    def reset(self) -> None: 
        """
        reset environment with random wear, changes the state
        create random wear state
        """
        item_wear = []
        for item in self.items:
            item_wear.append(random.randint(0, item.threshold))
        
        for i,item in enumerate(self.items):
            item.wear = item_wear[i]
        
        self.state=State(self.items)        
    
    def initial_state(self) -> State:
        initial_item = deepcopy(self.items)
        for item, init_wear in zip(
            initial_item, self._initial_wears
        ):
            item.wear = init_wear
        return State(initial_item)
    
    def step(self, action) -> Tuple[State, float]:
        reward = 0
        if not IsDoNothing(action):
            if not IsAcceptable(action, self.state):  
                return self.state,-99999                                  
            for item in self.items:
                item.improve(action[item])
                item.wearing_step()
                     
            self.state=State(self.items)
            reward += self.reward(action, self.state)

        else:# to avoid iterating over action
            for item in self.items:
                item.wearing_step()

            self.state=State(self.items)
            reward += self.reward(action, self.state)

        self.last_reward = reward
        
        return self.state, reward

    def reward(self, action, state):
        """
        calculate the reward R(action, state), known that the action is feasible
        """
        nb_preventive,nb_corrective=count_prev_corr(action)
        rew = 0
        for item in state.items:
            rew += item.productivity() # -0.5 is working great, 0.6 is more risky but works also fine
        if nb_corrective+nb_preventive>0:
            rew -= self.ship_cost
        rew -= nb_corrective * self.corr_cost
        rew -= nb_preventive * self.prev_cost
    
        return rew

    
    def convert_action_list_dict(self, action_list):
        """
        convert an action from a list to a dictionary
        """
        dic = dict()
        i = 0
        for item in self.items:
            dic[item] = action_list[i]
            i += 1
        return dic
    
 
    def getEveryState(self) -> List[State]:

        max_prods = [item.max_prod for item in self.items]
        thresholds = [item.threshold for item in self.items]
        wearing_func = self.items[0].wearing_func
        return State.get_states(
            max_prods=max_prods, thresholds=thresholds, wearing_func=wearing_func
        )
    def get_random_state(self):
        return random.choice(self.getEveryState())
    """    
    def getListState(self) -> List[int]:
        return [item.wear / item.threshold for item in self.items]

    def get_state_with_wear(self, wear: List[float]) -> State:
        self.new_items = deepcopy(self.items)
        for i,item in enumerate(self.new_items):
            item.wear = wear[i]
        return State( self.new_items)
    """

    @staticmethod
    def from_list(
        max_prods: List[float],
        thresholds: List[float],
        wearing_func: Callable,
    ) -> "Environnement":
        items = [
            Item(
                i,
                max_prod=max_prod,
                threshold=threshold,
                wearing_func=wearing_func,
            )
            for i, (max_prod, threshold) in enumerate(zip(max_prods, thresholds))
        ]
        return Environnement(items)
    
    def next_state(self, action:dict):
        env=deepcopy(self)
        env.step(action)
        return env.state


    @classmethod
    def from_floats(
        self,
        nb_items: int,
        max_prod: float,
        threshold: float,
        wearing_func: Callable,
    ) -> "Environnement":
        max_prods = nb_items * [max_prod]
        thresholds = nb_items * [threshold]
        return self.from_list(max_prods, thresholds, wearing_func)
    
    @classmethod
    def init(self, execution_type: str):
        execution_properties = execution(execution_type)
        return self.from_list(
            execution_properties[0],
            execution_properties[1],
            execution_properties[2],
        )