# Homework 1

Build a Linear Regression Model for a given dataset using Gradient Descent  
(NumPy and data visualization packages are allowed.)  
(Existing Linear Regression Model from any library/package is NOT allowed!)

1.	Import numpy and matplotlib.pyplot first in your python code 
Installation could be done by pip3  (e.g. “pip3 install numpy”)
e.g.
```
import numpy as np
import matplotlib.pyplot as plt
```

2.	**Generate data samples**: generate 10 pair of data samples that satisfying  
Y=2*X + 50 + a small random number   
e.g.
```
import numpy as np
x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2*x_data+50+5*np.random.random()
```
X could be within any range, stored as an array (using numpy.array)

3.	**Plot the landscape of the loss function.**  (2pts)
For example, the following code will print the landscape of Z[i][j]. Z is the loss function value of different ww and bb values.
X_data and y_data are the arrays of your samples.
e.g.
```
bb = np.arange(0,100,1) #bias
ww = np.arange(-5, 5,0.1) #weight
Z = np.zeros((len(bb),len(ww)))
 
for i in range(len(bb)):
    for j in range(len(ww)):
        b = bb[i]
        w = ww[j]
        Z[j][i] = 0        
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (w*x_data[n]+b - y_data[n])**2 # this is the loss 
        Z[j][i] = Z[j][i]/len(x_data)
```        

4.	Build a linear regression model that minimizing the loss for the given dataset using “gradient descent” algorithm introduced in lecture2.     (4pts)  
w<sub>j</sub>:=w<sub>j</sub>-α∙dL/dw  
Randomly pick some weights to start the “gradient descent” process. 
e.g.
```
b = 0 # initial b
w = 0 # initial w
```
**Explain how your gradient descent process was terminated (e.g. by testing convergence or finishing certain number of iterations) and explain all threshold values you used in your report.**    (1pt)
 
5.	Test different values of the learning rate “lr” and different number of iterations/convergence threshold values. 
**Explain how these values affect your program in your report.  (1pts)**
e.g.
```
lr = 0.0001 # example learning rate
iteration = 10000 # example iteration number
```

6.	**Track the change of the weight values (w and b) from each iteration and plot all the values out.  (2pts)**
e.g.
```
# Store parameters for plotting
b_history = [b]
w_history = [w]
# model by gradient descent
#…
b_history.append(b)
w_history.append(w)
#...
plt.plot(b_history, w_history, 'o-', ms=3, lw=1.5,color='black')
```

Example track change figure:  
![Track Change](Picture1.tif) 

7.	**Bonus** (up to 2pts)
**Compare the prediction result using your model with the given target values (Y values)
Or 
Any other type of model performance testing.**
