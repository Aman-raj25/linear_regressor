import numpy as np
import matplotlib.pyplot as plt

# Function to compute the error (Mean Squared Error) for a given line
def compute_error(b, m, points):
    totalError = 0  # Initialize total error as 0
    for i in range(len(points)):  # Iterate through all data points
        x = points[i, 0]  # Extract X value of the point
        y = points[i, 1]  # Extract Y value of the point
        # Compute squared error for the current point
        totalError += (y - (m * x + b)) ** 2  
    return totalError / float(len(points))  # Return the mean squared error

# Function to perform one step of gradient descent
def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0  # Initialize gradient for b (y-intercept)
    m_gradient = 0  # Initialize gradient for m (slope)
    n = float(len(points))  # Number of data points

    for i in range(len(points)):  # Iterate through all points
        x = points[i, 0]  # Get X value
        y = points[i, 1]  # Get Y value
        # Compute partial derivatives of the error function
        b_gradient += -(2 / n) * (y - ((m_current * x) + b_current))  
        m_gradient += -(2 / n) * x * (y - ((m_current * x) + b_current))  

    # Update b and m using gradient descent
    new_b = b_current - (learning_rate * b_gradient)  
    new_m = m_current - (learning_rate * m_gradient)  

    return [new_b, new_m]  # Return updated values

# Function to run the gradient descent algorithm
def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b  # Initialize b (y-intercept)
    m = starting_m  # Initialize m (slope)

    for i in range(num_iterations):  # Run for a given number of iterations
        b, m = step_gradient(b, m, np.array(points), learning_rate)  # Perform one step of gradient descent
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}: b = {b}, m = {m}, error = {compute_error(b, m, points)}")
        
        # Update visualization every 40 iterations, but only up to iteration 400
        if i % 40 == 0 and i <= 400:
            visualize(points, b, m, i)

    return [b, m]  # Return final values of b and m

# Function to visualize the regression process
def visualize(points, b, m, iteration):
    plt.clf()  # Clear the previous plot to update it dynamically

    x_vals = points[:, 0]  # Extract X values
    y_vals = points[:, 1]  # Extract Y values
    regression_line = m * x_vals + b  # Compute predicted Y values

    plt.scatter(x_vals, y_vals, color='blue', label="Data Points")  # Plot original data points
    plt.plot(x_vals, regression_line, color='red', label=f"Iteration {iteration}")  # Plot regression line
    plt.xlabel("X")  # Label for X-axis
    plt.ylabel("Y")  # Label for Y-axis
    plt.title(f"Linear Regression (Iteration {iteration})")  # Title with iteration info
    plt.legend()  # Show legend
    plt.pause(0.5)  # Pause to visualize before updating the next iteration

# Main function to run the linear regression
def run():
    try:
        # Load dataset from CSV file
        points = np.genfromtxt('Data.csv', delimiter=',')
        
        # Check if the dataset is valid
        if points.size == 0:
            raise ValueError("The CSV file is empty or not formatted correctly.")
    except Exception as e:
        print(f"Error loading data: {e}")  # Print error if file loading fails
        return

    # Define hyperparameters
    learning_rate = 0.0001  # Step size for gradient descent
    initial_b = 0  # Initial guess for b (y-intercept)
    initial_m = 0  # Initial guess for m (slope)
    num_iterations = 5000  # Number of iterations for gradient descent

    # Print starting values
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error(initial_b, initial_m, points)}")
    
    # Train the model using gradient descent
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    # Print final values after training
    print(f"Ending point at b = {b}, m = {m}, error = {compute_error(b, m, points)}")

    # Final visualization after training
    visualize(points, b, m, num_iterations)
    plt.show()  # Keep the final plot open

# Entry point for the script
if __name__ == '__main__':
    plt.ion()  # Enable interactive mode for real-time updates
    run()  # Run the regression model
