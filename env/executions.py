from typing import Callable, Dict, Union, List, Tuple

from env.wearing_functions import wearing_function


def execution(cmd:str)-> Tuple[List[float],List[float],Callable]:
    
    # customisable execution command : "4simple" -> 4 items discrete 
    # "12advanced" -> 12 items discrete (advanced config)
    nb = ""
    i = 0
    while i<len(cmd) and cmd[i].isdigit():
        nb = nb + cmd[i]
        i += 1
    if nb == "":
        raise NameError("not starting with an integer")
    nb_items = int(nb)
    
    remaining = cmd[i:]
    if "advanced" in remaining:
        threshold,wearing_fun = wearing_function("discrete2")
        max_prod = 1.
        return (nb_items * [max_prod],nb_items * [threshold],wearing_fun)
    else:
        threshold,wearing_fun = wearing_function("discrete")
        max_prod = 1.
        return (nb_items * [max_prod],nb_items * [threshold],wearing_fun)

