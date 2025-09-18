import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import threading
import time
import importlib
import inspect
import SiriusModule  # our module with functions

def list_functions(module):
    """List all user-defined functions in a module."""
    funcs = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:  # skip imported ones
            funcs[name] = obj
    return funcs

def hot_reload(module_name="SiriusModule"):
    """Reload a module and return its functions."""
    module = importlib.import_module(module_name)
    importlib.reload(module)
    return list_functions(module)

def adc14(x, y, formulaChoice):
    plt.figure()
    title = "ADC VALUE"
    YLabel = "ADC"
    YMax = 0
    flag = 0
    flag2 = 0
    if formulaChoice == "3":
        adc10_11(x,y)
        return

    if formulaChoice == "4":
        adc_thermistance(x,y)
        return
    if formulaChoice == "1":
        for i in range(len(y)):
            if y[i] > YMax:
                YMax = y[i]
                #print(YMax)
            y[i] = ((((((y[i]-10) *3.3)/4096)/209)*5000)/(0.003*5))*(9.81/2.2)
        
            if(flag == 0 and y[i] > 2900):
                start = x[i]
                flag2 = 1
            if(flag2 == 1 and y[i] < 1500):
                stop = x[i]
                flag2 = 0

        title = "THRUST"
        YLabel = "NEWTON"
    if formulaChoice == "2":
        for i in range(len(y)):
            if y[i] > YMax:
                YMax = y[i]
            y[i] = ((((((y[i]) *3.3)/4096)/209)*200)/(0.003*5))
        title = "TANK"
        YLabel = "LBS"

    #y = np.convolve(y, signal.firwin(4,500, window = np.hanning(len(x)), pass_zero = 'lowpass', fs = 1200), mode = 'same')
    print("burn time ish: " + f"{(stop - start)/1000000}")
    #plot.plot(x, np.real(y)) # np.real() uniquement pour éviter le "warning de casting complex values 
    plt.plot(x, y)
    plt.title(title + f"- MAX : {YMax}" +  f"- MAX Newton : {np.max(y)}")
    plt.xlim(start, stop)
    plt.xlabel("TimeStamp (ms)")
    plt.ylabel(YLabel)
    plt.show(block=False)

def main():
    data = {}

    with open(input("CSV path : "), "r") as din:
        lines = din.readlines()
        rem = input("REMOVE LAST COL? (y, n): ")
        header = []
        if rem == "y":
            header = lines[0].split(',')[0:-1]
        else:
            header = lines[0].split(',')
            header[-1] = header[-1][0:-1]
            
        for h in header:
            data[h.strip()] = []
        
        print("CONVERTING DATA...")
        
        for i in range(1, len(lines)):
            line = []
            if rem == 'y':
                line = lines[i].split(',')[0:-1]
            else:
                line = lines[i].split(',')
            if len(line) != len(header):
                print(f"ERROR : LINE {i} IGNORED")
                continue

            for l in range(len(header)):
                try:
                    data[header[l].strip()].append(int(line[l]))
                except:
                    data[header[l].strip()].append(line[l])
        
        print("---------DATA COMPLETED !!!----------------")
        print("Headers available : ", "|".join(header),end="|")
        return data
        
            
if __name__ == "__main__":
    data = main()
    while True:
        funcs = hot_reload("SiriusModule")
        print("\nAvailable functions:", list(funcs.keys()))

        choice = input("Enter function name (or 'quit'): ").strip()
        if choice == "quit":
            break
        if choice in funcs:
            try:
                chooseY = input("Choose data Y : ")
                funcs[choice](data[list(data.keys())[0]], data[chooseY], data)
            except Exception as e:
                print("Error:", e)
        else:
            print("Function not found.")