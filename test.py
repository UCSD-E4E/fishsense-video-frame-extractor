import glob

import cv2
import numpy as np
from skimage import metrics

f = list(glob.glob("/Users/aghos/Downloads/GX030308/*.jpg"))
imageNDArrays = {}
for file in f[:3]:
    imageNDArrays[file] = cv2.imread(file)

print("Image array is of size: ", len(imageNDArrays))

allPairs = [(a, b) for idx, a in enumerate(f) for b in f[idx + 1 :]]
print("All Pairs are of size: ", len(allPairs))
ssimScores = []
for pair in allPairs[:2]:
    a = imageNDArrays[pair[0]]
    b = imageNDArrays[pair[1]]

    # Resize
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)

    # Check structural similarity score
    ssim_score = metrics.structural_similarity(a, b, channel_axis=2)
    ssimScores.append(ssimScores)

    if ssim_score > 0.8:
        print(pair)

print("On average: ", np.average(np.array(ssimScores)))
