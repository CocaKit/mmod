import math
import random
from copy import copy 
import matplotlib.pyplot as plt

def teoretical(liam, mu, nu, nu_tw, gam):
    final_array = getFinalArray(liam, mu, nu, nu_tw, gam)
    Q, A = getSpecifications(final_array, liam, mu, nu)
    print(final_array)
    print(Q)
    print(A)
    return final_array

def getFinalArray(liam, mu, nu, nu_tw, gam):
    p_0 = 1 + (liam / (mu + nu)) + ((liam * nu + mu * nu_tw + nu * nu_tw)/(gam * (mu + nu)))
    p_0 = 1 / p_0
    p_1 =  p_0 * (liam / (mu + nu))
    p_2 = p_0 * (liam * nu + mu * nu_tw + nu * nu_tw)/(gam * (mu + nu))
    return [p_0, p_1, p_2]

def getSpecifications(final_array, liam, mu, nu):
    Q = final_array[0] * (mu / (mu + nu))
    A = liam * Q
    return Q, A

def modeling(liam, mu, nu, nu_tw, gam, total_time):
    #station - [name, start_time, end_time, start_broke, broke]
    rejected_array = []
    success_array = []
    stations_array = []

    time_req = 0
    time_break_out = getRandomTime(nu_tw)
    name_req = 1
    while time_req < total_time:
        #print(str(name_req) + " - " + str(time_req))
        #print("broke" + " - " + str(time_break_out))
        #broke in proceses
        if len(stations_array) != 0 and stations_array[-1][4] != 1 and stations_array[-1][2] < time_req:
            time_break_in = stations_array[-1][1] + getRandomTime(nu)
            if time_break_in < stations_array[-1][2]:
                success_array.remove(stations_array[-1][0])
                rejected_array.append(stations_array[-1][0])
                stations_array[-1][2] = time_break_in + getRandomTime(gam)
                stations_array[-1][3] = time_break_in
                stations_array[-1][4] = 1

        #requst early than break
        if time_req < time_break_out:
            #request to station
            if len(stations_array) == 0 or stations_array[-1][2] <= time_req:
                stations_array.append([name_req, time_req, time_req + getRandomTime(mu), 0, 0])
                success_array.append(name_req)
            else:
                rejected_array.append(name_req)
            name_req += 1
            time_req += getRandomTime(mu)
        else:
            #broke to station
            if len(stations_array) == 0 or stations_array[-1][2] <= time_break_out:
                stations_array.append([-1, time_break_out, time_break_out + getRandomTime(gam), time_break_out, 1])
            time_break_out += getRandomTime(nu_tw)

    """
    print("Success; " + str(success_array) + " - " + str(len(success_array)))
    print("Reject: " + str(rejected_array) + " - " + str(len(rejected_array)))
    print(stations_array)
    """

    print("Success; " + " - " + str(len(success_array)))
    print("Reject: " + " - " + str(len(rejected_array)))
    p_array = getP(stations_array, total_time)
    print(p_array)
    return p_array
            
def getRandomTime(var):
    return -1 * math.log(random.random()) / var

def getP(stations_array, hours):
    p_array = [0, 0, 0]
    var = 0.01
    while var < hours:
        changed = False
        for i in range(len(stations_array)):
            if stations_array[i][4] == 1:
                if stations_array[i][3] < var and stations_array[i][2] > var:
                    p_array[2] += 1
                    changed = True
                    break 
                elif stations_array[i][1] < var and stations_array[i][3] > var:
                    p_array[1] += 1
                    changed = True
                    break 
            else:
                if stations_array[i][1] < var and stations_array[i][2] > var:
                    p_array[1] += 1
                    changed = True
                    break 
        if not changed:
            p_array[0] += 1
        var += 0.01
    return [i / sum(p_array) for i in p_array]

def final_plots(final_array_station_teor, final_array_station_model):
    col_station_array = [1, 2, 3]
    plt.title('Stations')
    plt.plot(col_station_array, final_array_station_teor, 'b', col_station_array, final_array_station_model, 'r')
    plt.show()

def stable_plot(liam, mu, nu, nu_tw, gam, hours):
    station_matrix = [[] for i in range(3)]
    for i in range(1, hours + 1):
        final_array_station_model = modeling(liam, mu, nu, nu_tw, gam, i)
        for j in range(3):
            station_matrix[j].append(final_array_station_model[j])

    fig, ax = plt.subplots()
    for i in range(len(station_matrix)):
        s = 'st. p: ' + str(i)
        ax.plot([v for v in range(1, hours + 1)], station_matrix[i], label=s)
    ax.legend()
    plt.show()

print("##################################")
print("Teoretical:")
final_array_station_teor = teoretical(10, 6, 3, 4, 2)
print("##################################")
print("Modeling:")
final_array_station_model = modeling(10, 6, 3, 4, 2, 100)
print("##################################")

final_plots(final_array_station_teor, final_array_station_model)
stable_plot(15, 7, 3, 4, 2, 100)
