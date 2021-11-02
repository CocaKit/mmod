from random import *
import sys
import numpy as np
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
XI_CRIT_VAL = 3.841

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
    return emp_mat

def findCell(matrix, val):
    temp = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            temp += matrix[i][j]
            if val <= temp:
                return i, j

def buildPlot(rand_matrix, emp_mat):
    rand_matrix_one = np.reshape(rand_matrix, (1, len(Y_ARR) * len(X_ARR)))[0]
    emp_mat_one = np.reshape(emp_mat, (1, len(Y_ARR) * len(X_ARR)))[0]

    rand_matrix_one_str = [str(i) for i in rand_matrix_one]

    fig, ax = plt.subplots()

    ax.bar(rand_matrix_one_str, emp_mat_one)
    fig.set_figwidth(12)
    fig.set_figheight(6)    
    plt.show()

def accurMathExp(matrix, arr, ch):
    temp = 0
    if ch == "y":
        for i in range(len(arr)):
            temp += sum(matrix[i]) * arr[i]
    else:
        for i in range(len(arr)):
            str_sum = 0
            for j in range(len(matrix)):
                str_sum += matrix[j][i]
            temp += str_sum * arr[i]
    return temp / SAMPLE

def accurVari(arr, math_exp_arr):
    return sum([(var - math_exp_arr) ** 2 for var in arr])/len(arr)

def intervalMathExp(m, d):
    temp = STUDENT_VAL * sqrt(d / SAMPLE)
    return [m - temp, m + temp]

def intervalVari(d):
    temp = (SAMPLE - 1) * d 
    return [temp / XI_VAL_1, temp / XI_VAL_2]

def coeffCov(m_x, m_y, d_x, d_y):
    cov = sum([val - m_x for val in X_ARR]) * sum([val - m_y for val in Y_ARR])
    return cov / sqrt(len(X_ARR) * len(Y_ARR) * d_x * d_y)

def critPirs(t_mat, e_mat):
    crit_pirs_y = 0
    crit_pirs_x = 0
    for k in range(len(Y_ARR)):
        crit_pirs_y += ((sum(t_mat[k]) - (sum(e_mat[k]) / SAMPLE)) ** 2) / (sum(e_mat[k]) / SAMPLE)

    for i in range(len(X_ARR)):
        str_sum_t = 0
        str_sum_e = 0
        for j in range(len(Y_ARR)):
            str_sum_t += t_mat[j][i]
            str_sum_e += e_mat[j][i]
        crit_pirs_x += ((str_sum_t - (str_sum_e / SAMPLE)) ** 2) / (str_sum_e / SAMPLE)
    return crit_pirs_x, crit_pirs_y

def checkPirs(crit_pirs):
    return crit_pirs < XI_CRIT_VAL
         
rand_matrix = genRandMat()
emp_mat = genEmpMat(rand_matrix)
print("Theoretical matrix:")
print(rand_matrix)
print("Empirical matrix:")
print(emp_mat)

buildPlot(rand_matrix, emp_mat)

x_accur_math_exp = accurMathExp(emp_mat, X_ARR, "x")
y_accur_math_exp = accurMathExp(emp_mat, Y_ARR, "y")
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

pirs_x, pirs_y = critPirs(rand_matrix, emp_mat)
print("Pearson test x,y:", pirs_x, pirs_y)
check_x = checkPirs(pirs_x)
check_y = checkPirs(pirs_y)
print("there is no reason to reject the hypothesis x,y:", check_x, check_y)
