
import numpy as np
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

def compute_error(a, b, points):
    '''
        Computes Error = 1/N * sum((y - (ax + b))^2)
    '''
    N = len(points)
    x = points[:,0]
    y = points[:,1]

    error = sum((y - (a * x + b)) ** 2)
    return error / N

def gradient_descent(starting_a, starting_b, points, learning_rate, num_iterations):
    '''
        Performs gradient step num_iterations times
        in order to find optimal a, b values
    '''
    a, b = starting_a, starting_b
    for i in range(num_iterations):
        a, b = gradient_step(a, b, points, learning_rate)
    return a, b

def gradient_step(current_a, current_b, points, learning_rate):
    '''
        Updates a and b in antigradient direction
        with given learning_rate
    '''
    N = len(points)
    a, b = current_a, current_b

    x=points[:,0]
    y=points[:,1]

    a_grad=-(2/N)*(y-(a*x+b))*x
    b_grad=-(2/N)*(y-(a*x+b))

    a = current_a-learning_rate*sum(a_grad)
    b = current_b-learning_rate*sum(b_grad)
    return a, b

def run():
    # Step # 1 - Extract data
    points = np.genfromtxt('data.csv', delimiter=",")

    # Step # 2 - Define hyperparameters

    ## Learning rate
    learning_rate = 0.0001

    ## Coefficients y = a * x + b
    init_a = 0
    init_b = 0

    ## number of iterations
    num_iterations = 10000

    # Step 3 - model training

    print(
        'Start learning at a = {0}, b = {1}, error = {2}'.format(
            init_a,
            init_b,
            compute_error(init_a, init_b, points)
        )
    )

    a, b = gradient_descent(init_a, init_b, points, learning_rate, num_iterations)

    print(
        'End learning at a = {0}, b = {1}, error = {2}'.format(
            a,
            b,
            compute_error(a, b, points)
        )
    )
    return a, b

def main():
    a, b = run()


if __name__ == "__main__":
    main()
