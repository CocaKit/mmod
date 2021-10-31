from random import *
import sys
import numpy as np
import matplotlib.pyplot as plt

ST = 2
COL = 2
SAMPLE = 100
X_ARR = [1, 2]
Y_ARR = [3, 4]

def toFixed(numObj, digits=0):
    return float(f"{numObj:.{digits}f}")

def genRandMat():
    mat = np.array(np.random.dirichlet(np.ones(ST * COL)))
    return np.reshape(mat, (ST, COL))

def genEmpMat(matrix):
    emp_mat = np.zeros((ST, COL))
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
    rand_matrix_one = np.reshape(rand_matrix, (1, ST * COL))[0]
    emp_mat_one = np.reshape(emp_mat, (1, ST * COL))[0]

    rand_matrix_one_str = [str(i) for i in rand_matrix_one]

    fig, ax = plt.subplots()

    ax.bar(rand_matrix_one_str, emp_mat_one)
    fig.set_figwidth(12)
    fig.set_figheight(6)    
    plt.show()

def accurMathExp(arr):
    return sum(arr)/SAMPLE

def accurVari(arr, math_exp_arr):
    return sum([(var - math_exp_arr) ** 2 for var in arr])/SAMPLE

rand_matrix = genRandMat()
emp_mat = genEmpMat(rand_matrix)

#buildPlot(rand_matrix, emp_mat)

x_accur_math_exp = accurMathExp(X_ARR)
y_accur_math_exp = accurMathExp(Y_ARR)
print(x_accur_math_exp, y_accur_math_exp)
x_accur_var = accurVari(X_ARR, x_accur_math_exp)
y_accur_var = accurVari(Y_ARR, y_accur_math_exp)
print(x_accur_var, y_accur_var)
