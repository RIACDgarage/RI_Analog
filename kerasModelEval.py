import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

q1net = tf.keras.models.load_model("q1net_model")
q2net = tf.keras.models.load_model("q2net_model")

sample = 10000
# limit the max design, original is 1000
d1max = 300
d2max = 1000

# generate uniform random design in log space
design1 = np.random.uniform(low=np.log(3), high=np.log(d1max), size=sample)
design2 = np.random.uniform(low=np.log(3), high=np.log(d2max), size=sample)
# scale design to value 0 to 1 
design1 = design1/np.log(d1max)
design2 = design2/np.log(d2max)
design = np.array([design1, design2])
design = design.T

# make prediction from trained Keras models
pred1 = q1net.predict(design) * 100
pred2 = q2net.predict(design) * 100
    
design[:,0] = np.exp(design[:,0]*np.log(d1max))
design[:,1] = np.exp(design[:,1]*np.log(d2max))
x = design[:,0]
y = design[:,1]

# plot predict
#plt.figure()
plt.subplot(2, 2, 2)
plt.title("q1net model predict")
plt.xscale("log")
plt.yscale("log")
plt.scatter(x, y, c=pred1, cmap='inferno')
plt.colorbar()
plt.grid()
#plotname = "q1net.png"
#plt.savefig(plotname)

#plt.figure()
plt.subplot(2, 2, 4)
plt.title("q2net model predict")
plt.xscale("log")
plt.yscale("log")
plt.scatter(x, y, c=pred2, cmap='inferno')
plt.colorbar()
plt.grid()
#plotname = "q2net.png"
#plt.savefig(plotname)

# plot spice
df = pd.read_csv('spiceHist.csv')
df0 = df.loc[(df['design1'] <= d1max) & (df['design2'] <= d2max)]
xs = df0["design1"].to_numpy()
ys = df0["design2"].to_numpy()
m1 = df0["merit1"].to_numpy()
m2 = df0["merit2"].to_numpy()

#plt.figure()
plt.subplot(2, 2, 1)
plt.title("spice merit1")
plt.xscale("log")
plt.yscale("log")
plt.scatter(xs, ys, c=m1, cmap='inferno')
plt.colorbar()
plt.grid()
#plotname = "merit1.png"
#plt.savefig(plotname)

#plt.figure()
plt.subplot(2, 2, 3)
plt.title("spice merit2")
plt.xscale("log")
plt.yscale("log")
plt.scatter(xs, ys, c=m2, cmap='inferno')
plt.colorbar()
plt.grid()
#plotname = "merit2.png"
#plt.savefig(plotname)

plt.suptitle("Keras Model Prediction vs. Spice")
plotname = "kerasModelEval.png"
plt.savefig(plotname)

