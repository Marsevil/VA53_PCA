import os
import re
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

# Get list of files.
FILTER = re.compile("^[^\.]")
path = r"DataSet"
files = os.listdir(path)
# Filter files
files = list(filter(FILTER.match, files))


# Initialise list of image & average image.
delta = []
psi = np.zeros((4000000, 1))

# Iterate over files
for file in files:
    try:
        img = Image.open(path+'/'+file).convert('L')
    except (FileNotFoundError, UnidentifiedImageError):
        print(file, "Can't be opened as an image")
        continue
    
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

