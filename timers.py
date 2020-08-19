import time
import numpy as np

G_TIMERS = {}
G_TM_BEGINS = {}

def TM_CLEAR():
    G_TIMERS = {}

def TM_DECLEAR(name):
    if name not in G_TIMERS:
        G_TIMERS[name] = []

def TM_BEGIN(name):
    G_TM_BEGINS[name] = time.time()

def TM_END(name):
    t = time.time()
    TM_DECLEAR(name)
    G_TIMERS[name].append(t - G_TM_BEGINS.pop(name, t))

def TM_DISPLAY(remove_first_one=True):
    for key, values in G_TIMERS.items():
        startPos = 0
        if remove_first_one:
            startPos = 1
            if len(values) <= 1:
                continue
        print("{}:{} times, cost time avg:{}".format(key, len(values), np.average(values[startPos:])))

def TM_PICK(name):
    if name not in G_TIMERS or len(G_TIMERS[name])==0:
        return None
    else:
        return G_TIMERS[name][-1]


