from random import *
import sys
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from math import sqrt

ST = 4
COL = 4
SAMPLE = 600
X_ARR = [1, 2, 3, 4]
Y_ARR = [3, 4, 5, 6]
EPS = 0.95 #0.05
STUDENT_VAL = 1.647
XI_VAL_1 = 658.1
XI_VAL_2 = 544.2

def toFixed(numObj, digits=0):
    return float(f"{numObj:.{digits}f}")

def genRandMat():
    mat = np.array(np.random.dirichlet(np.ones(len(Y_ARR) * len(X_ARR))))
    return np.reshape(mat, (len(Y_ARR), len(X_ARR)))

def genEmpMat(matrix):
    emp_mat = np.zeros((len(Y_ARR), len(X_ARR)))
    for k in range(SAMPLE):
        val = toFixed(uniform(0, 1.00000), 5)
        st, col = findCell(matrix, val)
        emp_mat[st][col] += 1
    for i in range(len(emp_mat)):
        for j in range(len(emp_mat[i])):
            emp_mat[i][j] /= SAMPLE
    return emp_mat

def findCell(matrix, val):
    temp = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            temp += matrix[i][j]
            if val <= temp:
                return i, j

def buildPlot(rand_matrix, emp_mat):
    x_prob_rand_arr = []
    for i in range(len(X_ARR)):
        x_prob_rand_arr.append(sum(rand_matrix[:,i])) 

    y_prob_rand_arr = []
    for j in range(len(Y_ARR)):
        y_prob_rand_arr.append(sum(rand_matrix[j])) 

    x_prob_emp_arr = []
    for i in range(len(X_ARR)):
        x_prob_emp_arr.append(sum(emp_mat[:,i])) 

    y_prob_emp_arr = []
    for j in range(len(Y_ARR)):
        y_prob_emp_arr.append(sum(emp_mat[j])) 

    fig, ax = plt.subplots(2, 2)

    ax[0][0].bar(X_ARR, x_prob_rand_arr)
    ax[0][0].set_title("Probality x(teor)")

    ax[0][1].bar(Y_ARR, y_prob_rand_arr)
    ax[0][1].set_title("Probality y(teor)")

    ax[1][0].bar(X_ARR, x_prob_emp_arr)
    ax[1][0].set_title("Probality x(emp)")

    ax[1][1].bar(Y_ARR, y_prob_emp_arr)
    ax[1][1].set_title("Probality y(emp)")

    fig.set_figheight(8)
    fig.set_figwidth(14)

    plt.show()

def accurMathExp(matrix, arr, ch):
    temp = 0
    if ch == "y":
        for i in range(len(arr)):
            temp += sum(matrix[i]) * arr[i]
    else:
        for i in range(len(arr)):
            temp += sum(matrix[:,i]) * arr[i]
    return temp

def accurVari(arr, math_exp_arr):
    return sum([(var - math_exp_arr) ** 2 for var in arr]) / (len(arr) - 1)

def intervalMathExp(m, d):
    temp = STUDENT_VAL * sqrt(d / SAMPLE)
    return [m - temp, m + temp]

def intervalVari(d):
    temp = (SAMPLE - 1) * d 
    return [temp / XI_VAL_1, temp / XI_VAL_2]

def coeffCov(m_x, m_y, d_x, d_y):
    cov = sum([val - m_x for val in X_ARR]) * sum([val - m_y for val in Y_ARR])
    return cov / sqrt(len(X_ARR) * len(Y_ARR) * d_x * d_y * SAMPLE * SAMPLE)

def critPirs(t_mat, e_mat):
    crit_pirs_y = 0
    crit_pirs_x = 0
    for k in range(len(Y_ARR)):
        crit_pirs_y += ((sum(t_mat[k]) - sum(e_mat[k])) ** 2) / sum(e_mat[k])

    for i in range(len(X_ARR)):
        str_sum_t = 0
        str_sum_e = 0
        for j in range(len(Y_ARR)):
            str_sum_t += t_mat[j][i]
            str_sum_e += e_mat[j][i]
        crit_pirs_x += ((str_sum_t - str_sum_e) ** 2) / str_sum_e
    return crit_pirs_x, crit_pirs_y

def checkPirs(crit_pirs, n):
    return crit_pirs < scipy.stats.chi2.ppf(1 - EPS, df = n - 3)
         
def staticResearch(matrix):
    x_accur_math_exp = accurMathExp(matrix, X_ARR, "x")
    y_accur_math_exp = accurMathExp(matrix, Y_ARR, "y")
    print("Accurate math expection x,y:", x_accur_math_exp, y_accur_math_exp)

    x_accur_var = accurVari(X_ARR, x_accur_math_exp)
    y_accur_var = accurVari(Y_ARR, y_accur_math_exp)
    print("Accurate variation x, y:", x_accur_var, y_accur_var)

    interval_math_x = intervalMathExp(x_accur_math_exp, x_accur_var)
    print("Interval math expection x:", interval_math_x)
    interval_math_y = intervalMathExp(y_accur_math_exp, y_accur_var)
    print("Interval math expection y:", interval_math_y)

    interval_vari_x = intervalVari(x_accur_var)
    print("Interval variation x:", interval_vari_x)
    interval_vari_y = intervalVari(y_accur_var)
    print("Interval variation y:", interval_vari_y)

    coeff_cov = coeffCov(x_accur_math_exp, y_accur_math_exp, x_accur_var, y_accur_var)
    print("covariance coefficient:", coeff_cov)

rand_matrix = genRandMat()
emp_mat = genEmpMat(rand_matrix)
print("Theoretical matrix:")
print(rand_matrix)
print("Empirical matrix:")
print(emp_mat)

buildPlot(rand_matrix, emp_mat)

print("-------------------------------")
print("Teoretical matrix static research")
staticResearch(rand_matrix)

print("-------------------------------")
print("Emperical matrix static research")
staticResearch(emp_mat)

pirs_x, pirs_y = critPirs(rand_matrix, emp_mat)
print("-------------------------------")
print("Pearson test x,y:", pirs_x, pirs_y)
check_x = checkPirs(pirs_x, COL)
check_y = checkPirs(pirs_y, ST)
print("there is no reason to reject the hypothesis x,y:", check_x, check_y)
