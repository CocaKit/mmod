import math
import random
from copy import copy 
import matplotlib.pyplot as plt

def teoretical(n, m, liam, mu, nu):
    ro = liam / mu
    beta = nu / mu
    final_array_station, final_array_queue = getFinalArray(n, m, ro, beta)
    print(final_array_station)
    print(final_array_queue)
    L_queue, L_station, p_decline, Q, A, t_station = getSpecifications(final_array_station, final_array_queue, liam, mu, nu)
    return final_array_station, final_array_queue

def getSpecifications(final_array_station, final_array_queue, liam, mu, nu):
    L_queue, L_station = getAverage(final_array_station, final_array_queue)
    p_decline = 1 - (mu / liam) * L_station
    Q = 1 - p_decline
    A = liam * Q
    t_station = Q / mu + 1 / nu
    """
    print("Average requests in queue: " + str(L_queue))
    print("Average requests in all: " + str(L_station))
    print("P decline: " + str(p_decline))
    print("relative throughput: " + str(Q))
    print("absolute throughput: " + str(A))
    print("Average time request in all: " + str(t_station))
    """
    return L_queue, L_station, p_decline, Q, A, t_station 

def getFinalArray(n, m, ro, beta):
    p_0 = 1
    for j in range(1, n+1):
        p_0 += (ro ** j) / (math.factorial(j))

    temp = 0
    for i in range(1, m+1):
        temp_down = 1
        for l in range(1, i+1):
            temp_down *= n + l * beta
        temp += (ro ** i) / temp_down

    p_0 = 1 / (p_0 + ((ro ** n) / (math.factorial(n))) * temp)

    final_array_station = [0 for i in range(n+1)]
    final_array_queue = [0 for i in range(m)]

    final_array_station[0] = copy(p_0)

    for k in range(1, n+1):
        final_array_station[k] = p_0 * (ro ** k) / (math.factorial(k))

    for i in range(1, m + 1):
        temp_down = 1
        for l in range(1, i+1):
            temp_down *= n + l * beta
        
        final_array_queue[i-1] = final_array_station[-1] * (ro ** i) / temp_down

    return final_array_station, final_array_queue

def getAverage(final_array_station, final_array_queue):
    L_queue = 0
    for i in range(1, len(final_array_queue)+1):
        L_queue += i * final_array_queue[i-1]

    L_station = 0
    for i in range(0, len(final_array_station)):
        L_station += i * final_array_station[i-1]
    for i in range(1, len(final_array_queue)+1):
        L_station += len(final_array_station) * final_array_queue[i-1]

    return L_queue, L_station

def modeling(n, m, liam, mu, nu, total_time):
    #import pdb; pdb.set_trace()
    #requsts - [name, arrival_time]
    #station - [name, start_time, end_time]
    stations_array = [[] for i in range(n)]
    #queue - [name, start_time, end_time, decline_time, not_founded_station(0,1)]
    queuers_array = [[] for i in range(m)]
    rejected_array = []
    success_array = []

    new_requset_idx = 0
    time = 0
    while True:
        #print(str(new_requset_idx + 1) + " - " + str(time))
        if time >= total_time:
            break

        #from queue to station
        for q_idx in range(len(queuers_array)):
            if len(queuers_array[q_idx]) != 0 and queuers_array[q_idx][-1][4] == 0 :
                station_idx_temp = -1
                new_time = 0
                for s_idx in range(len(stations_array)):
                    if len(stations_array[s_idx]) == 0 or stations_array[s_idx][-1][2] <= queuers_array[q_idx][-1][3]:
                        if len(stations_array[s_idx]) != 0:
                            new_time = copy(stations_array[s_idx][-1][2])
                        station_idx_temp = copy(s_idx)
                        queuers_array[q_idx][-1][2] = copy(new_time)
                        break
                if station_idx_temp != -1:
                    stations_array[station_idx_temp].append([queuers_array[q_idx][-1][0], new_time, new_time + getRandomTime(mu)])
                    success_array.append(stations_array[station_idx_temp][-1][0])
                else:
                    queuers_array[q_idx][-1][2] = copy(queuers_array[q_idx][-1][3])
                    rejected_array.append(queuers_array[q_idx][-1][0])
                queuers_array[q_idx][-1][4] = 1
                
        #from request to station
        station_idx_temp = -1
        for s_idx in range(len(stations_array)):
            if len(stations_array[s_idx]) == 0 or stations_array[s_idx][-1][2] <= time:
                station_idx_temp = copy(s_idx)
                break
        if station_idx_temp != -1:
            stations_array[station_idx_temp].append([new_requset_idx + 1, time, time + getRandomTime(mu)])
            success_array.append(stations_array[station_idx_temp][-1][0])
        else:
            #from request to queue
            queue_idx_temp = -1
            for q_idx in range(len(queuers_array)):
                if len(queuers_array[q_idx]) == 0 or queuers_array[q_idx][-1][4] == 1:
                    if len(queuers_array[q_idx]) != 0 and queuers_array[q_idx][-1][2] > time:
                        continue
                    queue_idx_temp = copy(q_idx)
                    break
            if queue_idx_temp != -1:
                queuers_array[q_idx].append([new_requset_idx + 1, time, 0, time + getRandomTime(nu), 0])
            else:
                rejected_array.append(new_requset_idx + 1)
                
        new_requset_idx += 1
        time += getRandomTime(liam)


    """
    print("Success; " + str(success_array) + " - " + str(len(success_array)))
    print("Reject: " + str(rejected_array) + " - " + str(len(rejected_array)))
    showArrays(stations_array, "Station: ")
    showArrays(queuers_array, "Queue: ")
    """
                
    print("Success; " + " - " + str(len(success_array)))
    print("Reject: " + " - " + str(len(rejected_array)))
    p_station_array, p_queue_array = getP(stations_array, queuers_array, total_time)
    L_queue, L_station, p_decline, Q, A, t_station = getSpecifications(p_station_array, p_queue_array, liam, mu, nu)
    return p_station_array, p_queue_array

def getRandomTime(var):
    return -1 * math.log(random.random()) / var

def showArrays(var_array, info):
    for i in var_array:
        print(info + str(i))

def getP(stations_array, queuers_array, hours):
    all_array = stations_array + queuers_array
    p_array = [0 for i in range(len(all_array) + 1)]
    var = 0.01
    while var < hours:
        col = 0
        for st_idx in range(len(all_array)):
            for i in range(len(all_array[st_idx])):
                if all_array[st_idx][i][1] < var and all_array[st_idx][i][2] > var:
                    col += 1
                    break
        p_array[col] += 1
        var += 0.01
    sum_p = sum(p_array)
    p_array = [i / sum_p for i in p_array]
    return p_array[:len(stations_array) + 1], p_array[len(stations_array) + 1:]
       
def final_plots(final_array_station_teor, final_array_station_model, final_array_queue_teor, final_array_queue_model):
    col_station_array = [i for i in range(len(final_array_station_model))]
    col_queue_array = [i for i in range(len(final_array_queue_model))]
    plt.subplot(211)
    plt.title('Stations')
    plt.plot(col_station_array, final_array_station_teor, 'b', col_station_array, final_array_station_model, 'r')
    plt.subplot(212)
    plt.title('Queuers')
    plt.plot(col_queue_array, final_array_queue_teor, 'b', col_station_array, final_array_queue_model, 'r')
    plt.show()

def stable_plot(n, m, liam, mu, nu, hours):
    station_matrix = [[] for i in range(n + 1)]
    queue_matrix = [[] for i in range(m)]
    for i in range(1, hours + 1):
        final_array_station_model, final_array_queue_model = modeling(n, m, liam, mu, nu, i)
        for j in range(n + 1):
            station_matrix[j].append(final_array_station_model[j])
        for k in range(m):
            queue_matrix[k].append(final_array_queue_model[k])

    fig, ax = plt.subplots()
    for i in range(len(station_matrix)):
        s = 'st. p: ' + str(i)
        ax.plot([v for v in range(1, hours + 1)], station_matrix[i], label=s)
    for i in range(len(queue_matrix)):
        s = 'qu. p: ' + str(i)
        ax.plot([v for v in range(1, hours + 1)], queue_matrix[i], label=s)
    ax.legend()
    plt.show()

"""
print("##################################")
print("Teoretical:")
final_array_station_teor, final_array_queue_teor = teoretical(3, 4, 10, 3, 6)
print("##################################")
print("Modeling:")
final_array_station_model, final_array_queue_model = modeling(3, 4, 10, 3, 6, 3000)
print("##################################")
final_plots(final_array_station_teor, final_array_station_model, final_array_queue_teor, final_array_queue_model)
"""
stable_plot(3, 4, 10, 3, 6, 100)


