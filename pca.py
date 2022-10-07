import os
import re
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

# Define constant
FILTER = re.compile("^[^\.]")
TRAIN_PATH = r"DataSet/Train"
TEST_PATH = r"DataSet/Test"
EXPECTED_SIZE = (2000, 2000)
VECTOR_SIZE = EXPECTED_SIZE[0] * EXPECTED_SIZE[1]

# Get list of files.
files = os.listdir(TRAIN_PATH)
# Filter files
files = list(filter(FILTER.match, files))


# Initialise list of image & average image.
delta = []
fileNames = []
psi = np.zeros((4000000, 1))

# Iterate over files
for file in files:
    try:
        img = Image.open(os.path.join(TRAIN_PATH, file)).convert('L')
    except (FileNotFoundError, UnidentifiedImageError):
        print(file, "Can't be opened as an image")
        continue
    
    delta_i = np.asmatrix(img)

    if delta_i.shape != (2000, 2000):
        continue

    delta_i = delta_i.flatten().T
    delta.append(delta_i)
    fileNames.append(file)
        
    # Accumulate average image.
    psi += delta_i
    
# Process average image
psi /= len(delta)

# Initialise normalized image
phi = np.empty((4000000, 0))

for delta_i in delta:
    #phi.append(delta_i - psi)
    phi = np.c_[phi, delta_i - psi]

A = phi

# Process eigen vectors and value of covariance Matrix AAT through eigen values and vectors of ATA.
(lambda_i, v_i) = np.linalg.eig(A.T*A)
u_i = A * v_i

# Define passage matrix P as matrix from concatenate eigen vectors
P = u_i
Pi = np.linalg.pinv(P)

vecs = Pi * phi

# Test with an image.
testImagePath = os.path.join(TEST_PATH, r"IMG_2147_R.jpeg")
img = Image.open(testImagePath).convert('L')
img = np.asmatrix(img)
img = img.flatten().T
img = Pi * img
resultDistance = float('inf')
result = ""
for vecInd in range(vecs.shape[1]):
    vec = vecs[:, vecInd]
    distance = np.linalg.norm(vec - img)
    if distance < resultDistance :
        resultDistance = distance
        result = fileNames[vecInd]

print("Closed image is :", result, "With a distance of", resultDistance)
