import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import threading
import time
import SiriusUtils
import multiprocessing
from scipy.signal import savgol_filter, butter, filtfilt

def adc_denoise(x,y, *args):
    y = savgol_filter(y, window_length=1000, polyorder=3)

    maxY = 0
    print(np.array(y).max())
    for i in range(len(y)):
        y[i] = float(y[i])
        x[i] = float(x[i])
        
        if maxY < y[i]:
            maxY = y[i]
    
    plt.figure()
    plt.title(f"FILTERED ADC VALUES MAX : {maxY:.2f}")
    plt.plot(x, y)
    plt.show()


def adc_avg(x,y, *args):
    window_size = 64
    y_moving_avg = np.convolve(y, np.ones(window_size)/window_size, mode='same')
    plt.figure()
    plt.plot(x,y_moving_avg)
    plt.title("Moving Average")
    plt.show()


def adc_thermistance(x,y, *args):
    valMax=0
    y0 = SiriusUtils.y_filtered(args[0]["ADC_0"])
    y1 = SiriusUtils.y_filtered(args[0]["ADC_1"])
    y2 = SiriusUtils.y_filtered(args[0]["ADC_2"])
    y3 = SiriusUtils.y_filtered(args[0]["ADC_3"])
    y4 = SiriusUtils.y_filtered(args[0]["ADC_4"])
    y5 = SiriusUtils.y_filtered(args[0]["ADC_5"])
    y6 = SiriusUtils.y_filtered(args[0]["ADC_6"])
    y7 = SiriusUtils.y_filtered(args[0]["ADC_7"])
    list_of_y = [y0,y1,y2,y3,y4,y5,y6,y7]
    with multiprocessing.Pool() as pool:
        print("CALC IN PROGRESS...")
        result = pool.map(SiriusUtils.calcThermistance, list_of_y)
        percent = 0
        for i in range(len(y)):
            if valMax < y[i]:
                valMax = y[i]

            percent = i/len(y)*100
            if(percent % 25 == 0):
                print(f"\rPERCENT : {percent:.2f}", end="\r")
    
        print("saving data in a file!")
        t = threading.Thread(target=SiriusUtils.data_to_file, args=("thermistance.json", result))
        t.start()

        plt.plot(x,result[0], label="ADC_0")
        plt.plot(x,result[1], label="ADC_1")
        plt.plot(x,result[2], label="ADC_2")
        plt.plot(x,result[3], label="ADC_3")
        plt.plot(x,result[4], label="ADC_4")
        plt.plot(x,result[5], label="ADC_5")
        plt.plot(x,result[6], label="ADC_6")
        plt.plot(x,result[7], label="ADC_7")
        valMax = SiriusUtils.adc_to_temperature(valMax)
        plt.title(f"THERMISTANCE C MAX : {valMax:.2f}")
        plt.legend()
        plt.show(block = False)

def adc(x,y:list, *args):
    plt.figure()
    maxY = 0
    print(np.array(y).max())
    for i in range(len(y)):
        y[i] = float(y[i])
        x[i] = float(x[i])
        
        if maxY < y[i]:
            maxY = y[i]
    plt.title(f"ADC VALUES MAX : {maxY}")
    plt.plot(x,y)
    #plt.xlim(2.393e9, 2.403e9)
    plt.show()

def adc_trust(x, y, *args):
    plt.figure()
    startI = int(input("INDEX for start: "))
    stopI = int(input("INDEX for STOP: "))
    y = y[startI:stopI]
    x = x[startI:stopI]
    title = "ADC VALUE"
    YLabel = "ADC"
    YMax = 0
    flag = 0
    flag2 = 0
    formulaChoice = input("1 - thrust 2- tank : ")
    y = SiriusUtils.y_filtered(y)
    yadc = np.copy(y)
    start = 0
    stop = 0
    if formulaChoice == "1":
        y = ((((((y) *3.3)/4096)/209)*5000)/(0.003*5))*(9.81/2.2)
        title = "THRUST"
        YLabel = "NEWTON"
        #plt.xlim(start, stop)
    if formulaChoice == "2":
        y = ((((((y) *3.3)/4096)/209)*200)/(0.003*5))
        title = "TANK"
        YLabel = "LBS"

    #y = np.convolve(y, signal.firwin(4,500, window = np.hanning(len(x)), pass_zero = 'lowpass', fs = 1200), mode = 'same')
    
    #plot.plot(x, np.real(y)) # np.real() uniquement pour éviter le "warning de casting complex values 
    plt.plot(x, y)
    plt.title(title + f"- MAX Newton : {np.max(y)}")
    plt.xlabel("TimeStamp (us)")
    plt.ylabel(YLabel)
    plt.show()
    with open("Thrust.csv", "w") as file:
        file.write(f"Time [s](starting at {x[0]} us),PSI,ADC,\n")
        for i in range(len(x)):
            file.write(f"{((x[i]-x[0])/1e6):.4f},{y[i]},{yadc[i]},\n")
    #1.22 to 1.231

def adc_total(x,y, *args):
    plt.figure()
    for i in range(16):
        plt.plot(x,SiriusUtils.y_filtered(args[0][f"ADC_{i}"]), label=f"ADC_{i}")
    
    plt.title("ADC ALL")
    plt.legend()
    plt.show()

def adc_pt_chamber(x,y, *args):
    print("CHAMBER")
    startI = int(input("INDEX for start: "))
    y = y[startI:]
    x = x[startI:]
    yMax = 0
    y = SiriusUtils.y_filtered(y)
    yNp = np.array(y)

    newY = (yNp-807.62)/1.019
    yMax = newY.max()
    plt.figure()
    plt.title(f"PT CHAMBER MAX : {yMax:.2f}")
    plt.plot(np.array(x),newY)
    plt.show()

    with open("Chamber.csv", "w") as file:
        file.write(f"Time [s](starting at {x[0]} us),PSI,ADC,\n")
        for i in range(len(x)):
            file.write(f"{((x[i]-x[0])/1e6):.4f},{newY[i]},{y[i]},\n")

def find_index(x:list,y, *args):
    value = float(input("Enter the value X to find: "))

    # Find nearest value
    nearest = min(x, key=lambda n: abs(n - value))

    print(f"Nearest value to {value} is {nearest}")
    print("Index of the nearest value : ", x.index(nearest))

def export_csv(x,y,*args):
    with open("Export.csv", "w") as file:
        start = int(input("START X: "))
        stop = int(input("STOP X: "))

        file.write("timeStamp,Data,\n")
        for i in range(start,len(y)):
            if(i >= stop):
                break
            file.write(f"{x[i]},{y[i]},\n")
    
    print("EXPORT COMPLETED..")

def adc_pt(x,y, *args):
    print("PT in PSI")
    startI = int(input("INDEX for start: "))
    y = y[startI:]
    x = x[startI:]
    y = SiriusUtils.y_filtered(y)
    start = 0
    stop = 0
    flag2 = 0
    flag = 0
    maxVal = 0
    a = 0
    b = 0
    yNp = np.array(y)
    if input("CHOICE 1 : CHAMBER, 2: TANK :") == "1":
        a = 0.9438
        b = 43
    else:
        a = 0.9202
        b = -43

    newY = a*yNp+b

    print(start,stop)
    plt.plot(x,newY)
    plt.ylabel("PSI")
    #plt.xlim(7.445e7, 7.55e7)
    plt.title(f"PT- MAX : {newY.max():.2f}")
    plt.show()
