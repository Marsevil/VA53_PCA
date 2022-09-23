import os
import numpy as np
from PIL import Image

# Get list of files.
path = r"DataSet"
files = os.listdir(path)

# Initialise list of image & average image.
delta = []
psi = np.zeros((4000000, 1))

# Iterate over files
for file in files:
    img = Image.open(path+'/'+file).convert('L')
    
    delta_i = np.asmatrix(img)
    delta_i = delta_i.flatten().T
    delta.append(delta_i)
    
    psi += delta_i
    
# Process average image
psi /= len(delta)

# Initialise normalized image
phi = []

for delta_i in delta:
    phi.append(delta_i - psi)

A = np.asarray(phi)

print(A.shape)

#wAAT = np.linalg.eigvals(A*A.T)
#wATA = np.linalg.eigvals(A.T*A)

#print("wAAT =", wAAT)
#print("wATA =", wATA)

