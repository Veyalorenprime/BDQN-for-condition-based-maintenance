from typing import Callable, Tuple 
import numpy as np
from random import choices

def wearing_function(function_type:str) -> Tuple[float,Callable]:
    linked_functions = {
        "discrete": discrete_wearing_function(),
        "discrete2": discrete_wearing_function2(),
    }
    if not function_type in linked_functions.keys():
        raise NameError("wrong function name")
    return linked_functions[function_type]


def wearing_prob(i,j,p,q,s):      #probability to go from state i to j
    """
    probability to go from state i to state j
    s: total number of states
    p: probability to get to the next state of degradation
    q: probability to get to final state of degradation
    """
    if i==j:
        if i!=s-1:
            return 1-p-q #prob to stay in the state if not final state
        else:
            return 1 
    elif j==i+1:
        return p
    elif j==s-1 and i<s-2:
        return q

    else:
        return 0
# needs to be reviewd
def markov_matrice(s,p,q):          #Stochastique matrix associated to our Markov chain
    A=np.zeros((s,s))
    for i in range(s):
        for j in range(s):
            A[i][j]=wearing_prob(i,j,p,q,s)
    return A

def discrete_wearing_function()-> Tuple[float,Callable]:
    """ return threshold and function
    uses above defined markov matrix
    """
    def wearing_func(threshold,wear):
        p=0.1
        q=0.02
        s=threshold
        A=markov_matrice(s+1,p,q)
        w=wear
        L=A[w]
        list_probas=[]
        for i in range(len(L)):
            list_probas.append(L[i])
        print('L', L)
        print(list_probas)
        choix=choices(range(len(L)),weights=list_probas)
        wear=choix[0]
        return wear
    threshold = 5
    return threshold,wearing_func


def discrete_wearing_function2()-> Tuple[float,Callable]:
    """ return threshold and function
    uses below defined wearing matrix"""
    def wearing_func(wear):
        A=np.array([
    [0.9848, 0.015, 0, 0, 0.0002],
    [0, 0.989, 0.01, 0, 0.001],
    [0, 0, 0.98, 0.005, 0.015],
    [0, 0, 0, 0.94, 0.06],
    [0, 0, 0, 0, 1]
    ])
        w=wear
        L=A[w]
        list_probas=[]
        for i in range(len(L)):
            list_probas.append(L[i])
        choix=choices(range(len(L)),weights=list_probas)
        wear=choix[0]
        return wear
    threshold = 4 
    return threshold,wearing_func