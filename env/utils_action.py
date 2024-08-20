import numpy
def act_dict_to_list_array(action: dict, items:list):
    action_list=[]
    for item in items:
        action_list.append(action[item])
    return action_list,np.array(action_list)

def count_prev_corr(action:dict):
    """
    counts the number of preventive corrections and corrective ones"""
    val=list(action.values())
    nb_prev=val.count(1)
    nb_corr=val.count(2)
    return nb_prev, nb_corr

def IsDoNothing(action:dict):
    """determines if an action does nothing"""
    val=list(action.values())
    nb_nothing=val.count(0)
    return nb_nothing==len(val)

def IsAcceptable(action,state):
    poss_act=state.getPossibleActions()
    for item in action.keys():
        if item in state.items:
            if not (action[item] in poss_act[item]):
                return False 
    return True