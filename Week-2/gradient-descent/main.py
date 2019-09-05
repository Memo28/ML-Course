from numpy import *

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iteration):
    b = starting_b
    m = starting_m

    for i in range(num_iteration):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    #gradient calculating for the partial derivate 
    #https://miro.medium.com/max/1192/1*3YJx2rdqMW5ccRJZFH9v6w.png

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += - (2/N) * (y - ((m_current * x) + b_current))
        m_gradient += - (2/N) * x  *(y - ((m_current * x) + b_current))
    
    #Update b and m based in the gradient
    new_b = b_current - (learning_rate  * b_gradient)
    new_m = b_current - (learning_rate * m_gradient)

    return [new_b, new_m]

#This function will calculate the bias, the bias is the sum of square errors of the distance between a point 
#in the plot and the line of the function
def compute_error_for_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[0, i]

        #Equation https://miro.medium.com/max/1230/1*AQKoBlrYPA6kjvW8XomKUQ.png
        totalError = (y - (m * x + b)) **2
    return totalError / float(len(points))



def run():
    #Read the data from the csv, splited by delimiter ","
    points = genfromtxt('data.csv',delimiter=",")
    
    #How fast our model learns
    #If too low the model will be to slow to convert, if too high the model will never convert
    #Convert means find the right equation to solve the model with the minimuan bias and variance
    learning_rate = 0.0001

    #y = mx + b => slope formula
    initial_b = 0
    intial_m = 0
    num_iteration = 1000

    [b, m] = gradient_descent_runner(points, initial_b, intial_m, learning_rate, num_iteration)
    #Returns the ideal values for b and m and if we put those values in the equation
    #We are going to have the best line to fit our training plot data
    print(b)
    print(m)


if  __name__ == "__main__":
    run()