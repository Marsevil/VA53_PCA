import os
import re
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

# Define constant
FILTER = re.compile("^[^\.]")
PATH = r"DataSet"
EXPECTED_SIZE = (2000, 2000)

# Get list of files.
files = os.listdir(PATH)
# Filter files
files = list(filter(FILTER.match, files))


# Initialise list of image & average image.
delta = []
psi = np.zeros((4000000, 1))

# Iterate over files
for file in files:
    try:
        img = Image.open(PATH+'/'+file).convert('L')
    except (FileNotFoundError, UnidentifiedImageError):
        print(file, "Can't be opened as an image")
        continue
    
    delta_i = np.asmatrix(img)

    if delta_i.shape != (2000, 2000):
        continue

    delta_i = delta_i.flatten().T
    delta.append(delta_i)
        
    # Accumulate average image.
    psi += delta_i
    
# Process average image
psi /= len(delta)

# Initialise normalized image
phi = []

for delta_i in delta:
    phi.append(delta_i - psi)

A = np.asarray(phi)
A = np.squeeze(A, 2)
A = np.asmatrix(A)

(lambda_i, v_i) = np.linalg.eig(A*A.T)

