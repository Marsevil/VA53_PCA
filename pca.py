import os
import sys
import re
import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError

###################################
# ---------- Constants ---------- #
###################################
FILTER = re.compile("^[^\.]")
BASE_DIR = r"DataSet"
TRAIN_PATH = os.path.join(BASE_DIR, r"Train")
TEST_PATH = os.path.join(BASE_DIR, r"Test")
MATRIX_FILE = os.path.join(BASE_DIR, r"preProcessedData.npy")
EXPECTED_SIZE = (2000, 2000)
EXPECTED_VECTOR_SIZE = EXPECTED_SIZE[0] * EXPECTED_SIZE[1]


###################################
# ---------- Functions ---------- #
###################################
def readImageAsVector(filePath):
    # Open the image
    img = Image.open(filePath).convert('L')
    # Convert into numpy flat column vector
    img = np.asmatrix(img).flatten().T
    return img

def buildUncorrelatedBasis(inputVectors, minInformationalContrib):
    # Build a matrix from vectors
    A = np.empty((inputVectors[0].shape[0], 0))
    for vec in inputVectors:
        A = np.c_[A, vec]
    
    # Process eigen vals & vectors of ATA
    (lambda_i, v_i) = np.linalg.eig(A.T*A)
    # Process eigen vectors of AAT by multiplying by A ((A*A^T)*A*v_i = lambda_i*A*v_i)
    u_i = A*v_i

    # Passage matrix is concatenate eigen vectors
    P = u_i
    # Process P^-1
    Pi = np.linalg.pinv(u_i)

    return (P, Pi)


##############################
# ---------- Main ---------- #
##############################

# Get list of files.
files = os.listdir(TRAIN_PATH)
# Filter files
files = list(filter(FILTER.match, files))# Initialise list of image & average image.

delta = []
fileNames = []
psi = np.zeros((EXPECTED_VECTOR_SIZE, 1))

# Iterate over files
for file in files:
    try:
        delta_i = readImageAsVector(os.path.join(TRAIN_PATH, file))
    except (FileNotFoundError, UnidentifiedImageError):
        print(file, "Can't be opened as an image")
        continue

    # Test if img have the right size
    if delta_i.shape != (EXPECTED_VECTOR_SIZE, 1):
        continue

    delta.append(delta_i)
    fileNames.append(file)
        
    # Accumulate average image.
    psi += delta_i
    
# Process average image
psi /= len(delta)

# Normalize every loaded image
phi = []
for delta_i in delta:
    phi.append(delta_i - psi)

# Load or generate matrix
P = None
Pi = None
if os.path.exists(MATRIX_FILE):
    with open(MATRIX_FILE, "rb") as f:
        P = np.load(f)
        Pi = np.load(f)
else:
    (P, Pi) = buildUncorrelatedBasis(phi, None)
    with open(MATRIX_FILE, "wb") as f:
        np.save(f, P)
        np.save(f, Pi)

# Project all vectors in the new basis
projectedVecs = []
for vec in phi:
    projectedVecs.append(Pi * vec)

# Test with an image.
# Load image
testImagePath = sys.argv[1]
img = readImageAsVector(testImagePath)
# Project vector
img = Pi * img

# Search for the closest base vector
resultDistance = float('inf')
resultFileName = ""
for vecInd in range(len(projectedVecs)):
    vec = projectedVecs[vecInd]
    distance = np.linalg.norm(vec - img)
    if distance < resultDistance:
        resultDistance = distance
        resultFileName = fileNames[vecInd]

print("Closest image is :", resultFileName, "With a distance of", resultDistance)

