from typing import Callable

class Item:
    """Class that represents an item whose wearing steps following a continuous process.

    Args:
        id (int): The item's id.
        wearing_func (Callable): The function returning wearing steps.
        threshold (float): If the wear of the item is greater than the threshold, it is stopped.
        max_prod (float): The energy produced by the item at full capacity.
    """

    def __init__(
        self,
        id: int,
        threshold: float,
        wearing_func: Callable,
        wear: float = 0,
        max_prod: float = 1.,
    ) -> None:
        self.id = id
        self.wear = wear
        self.threshold = threshold
        self.wearing_func = wearing_func
        self.max_prod = max_prod

    def wearing_step(self) -> None:
        """
        moves to a state with a certain probability
        """
        if self.wear < self.threshold:
            self.wear = min(
                self.threshold,
                self.wearing_func(self.wear),
            )
    def improve(self, elementary_act: int) -> None:
        """
        apply maintenance on item
        """
        if elementary_act==0:
            pass
        elif elementary_act==1:
            self.wear=0
        elif elementary_act==2:
            self.wear=0
        else: 
            raise Exception("No valid action for item :"+ str(self.id))

    def productivity(self) -> float:
        """
        calculates the production (energy gain) of the item
        """
        return self.max_prod if self.wear < self.threshold else 0.0

    def reset(self) -> None: #set to a new state
        self.wear = 0
        
    def __eq__(self, other):
        return self.wear == other.wear and self.max_prod == other.max_prod and self.threshold == other.threshold
    
    def __hash__(self) -> int: ## 
        return self.id #,self.max_prod,self.threshold)
    
    def __str__(self) -> str:
        return str(self.wear)
    
    def __repr__(self) -> str:
        return self.__str__()
