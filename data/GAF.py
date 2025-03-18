from pyts.approximation import PiecewiseAggregateApproximation
from pyts.preprocessing import MinMaxScaler
import numpy as np 
import matplotlib.pyplot as plt
X = [[0.02457,0.03351,0.03398,0.05788,0.04248,0.06684,0.07574,0.07671,0.07763,0.08600,0.10814,0.10704,0.13024,0.1297,0.24823,0.29578,0.65176,1.00777,1.56948,1.89378,2.13114,2.39228,2.64538,2.93022,3.13589,3.31784,3.54725,3.79255,3.99033,4.15643,4.37797,4.57580,4.71031,4.72555,4.71625,4.71604,4.73936,4.73101],
      [0.61885,1.39588,1.75559,1.88523,2.20168,2.67664,3.42490,4.15871,4.86376,5.20913,4.00062,3.16608,2.76334,2.37485,2.24603,2.30386,2.27713,2.27918,2.28241,2.22673,2.24248,2.28715,2.24544,2.26147,2.24826,2.23492,2.22185,2.25204,2.25318,2.22536,2.24102,2.27094,2.28610,1.84015,0.78973,0.63146,0.31505,-0.01594 ]]
plt.plot(X[0],X[1])
plt.title("Time series")
plt.xlabel("timestamp")
plt.ylabel("value")
plt.show()
# PAA
transformer = PiecewiseAggregateApproximation(window_size=2)
result = transformer.transform(X)

#Scaling in interval [0,1]
scaler = MinMaxScaler()
scaled_X = scaler.transform(result)

arccos_X = np.arccos(scaled_X[1, :])
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(result[0,:], arccos_X)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2]) # Less radial ticks
ax.set_rlabel_position(-22.5) # Move radial labels away from plotted line
ax.grid(True)
ax.set_title("Polar coordinates", va="bottom")
plt.show()
field = [a+b for a in arccos_X for b in arccos_X]
gram = np.cos(field).reshape(-1,19)
plt.imshow(gram)
plt.show()
