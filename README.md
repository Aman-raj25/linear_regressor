# Linear Regression with Gradient Descent

This project implements **Linear Regression** using **Gradient Descent** in Python. The model is trained using a dataset from a CSV file and visualizes the training process at regular intervals.

## Features
✅ Implements **Batch Gradient Descent** for linear regression  
✅ **Real-time visualization** of the regression line every 200 iterations  
✅ **Mean Squared Error (MSE)** calculation for performance evaluation  
✅ Supports custom **learning rates and iteration counts**  
✅ **Error handling** for dataset loading  

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed along with the required libraries:
```sh
pip install numpy matplotlib
```

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/linear-regression.git
   cd linear-regression
   ```
2. Place your dataset in the project directory as `Data.csv`. The dataset should be in the format:
   ```csv
   X,Y
   1,2
   2,2.8
   3,3.6
   4,4.5
   ```
3. Run the program:
   ```sh
   python main.py
   ```

## Expected Output
- The console will display the updated values of `b`, `m`, and **error** at every 100 iterations.
- A **real-time visualization** will update every 40 iterations till iteration 400(you can change it, as per need), showing how the regression line improves.

## File Structure
```
├── Data.csv         # Input dataset
├── main.py          # Main script for training & visualization
├── README.md        # Project documentation
```

## Example Output
```
Starting gradient descent at b = 0, m = 0, error = 177.0
Iteration 100: b = 0.1, m = 1.5, error = 10.2
...
Ending point at b = 0.33, m = 2.09, error = 0.096
```

## License
This project is licensed under the **MIT License**.

## Contributing
Feel free to fork this repository and submit **pull requests**. For major changes, please open an issue first to discuss your ideas.

---
Happy Coding! 🚀

