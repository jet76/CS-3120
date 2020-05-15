import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([35., 38., 31., 20., 22., 25., 17., 60., 8., 60.])
y_data = 2*x_data+50+5*np.random.random()

bias = np.arange(0, 100, 1)  # bias
weight = np.arange(-5, 5, 0.1)  # weight
Z = np.zeros((len(bias), len(weight)))

for i in range(len(bias)):
    for j in range(len(weight)):
        bb = bias[i]
        ww = weight[j]
        Z[j][i] = 0
        for n in range(len(x_data)):
            Z[j][i] = Z[j][i] + (ww*x_data[n]+bb - y_data[n]
                                 )**2  # this is the loss

        Z[j][i] = Z[j][i]/len(x_data)

b = 0  # initial b
w = 0  # initial w
lr = 0.0001  # learning rate
iteration = 10000
b_history = [b]
w_history = [w]

for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad += w*x_data[n] + b - y_data[n]
        w_grad += (w*x_data[n] + b - y_data[n])*x_data[n]

    # Update parameters.
    b -= lr * b_grad
    w -= lr * w_grad
    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)

print('Predicted w = %s' % w)
print('Predicted b = %s' % b)

p_data = w*x_data+b

for n in range(len(p_data)):
    print('Predicted Y: %-*s Target Y: %-*s Difference: %s' %
          (20, p_data[n], 20, y_data[n], p_data[n] - y_data[n]))

plt.xlim(0, 100)
plt.ylim(-5, 5)
plt.xlabel('b = %s' % (b))
plt.ylabel('w = %s' % (w))
plt.figtext(.13, .90, 'lr = %s' % (lr))
plt.figtext(.75, .90, 'iter = %s' % (iteration))
plt.contourf(bias, weight, Z, 50, alpha=0.5, cmap=plt.get_cmap('jet'))
plt.plot(b_history, w_history, 'o-', ms=1.0, lw=1.5, color='black')
plt.plot(b, w, 'x', ms=10, c='red')
plt.show()
