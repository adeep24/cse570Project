# @author Aakash
# This work is done towards cse570 wireless project.

# necessary imports

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import math
import pprint as pp


filePath = "data/ScenarioA_omni.mat"
fileContent = sio.loadmat(filePath)

# fileContent is dictionary with following keys and corresponding values
# cir
# distance
# firstPath
# floorplan
# posBS
# posEP
# preamCnt


# for k,v in fileContent.items():
#   pp.pprint(k)
#   pp.pprint(v)
#   print()

# signal value at particular time
def getSDW(t=0.000001):
  R = 0.9
  Tp = 2.4 * 10 ** (-9)
  # Tp = 1

  r = 1
  i = 2

  left = (math.pi * t)/Tp

  realPart = math.sin(left)/left

  right = (math.pi * R * t) / Tp

  bottom = 1 - (((2 * R * t) / Tp) ** 2)
  imagPart = math.cos(right) / bottom
  # print(realPart, imagPart)
  a = np.complex128(complex(realPart, imagPart))
  # print(a)
  return a

# return point on circle define by (x,y) center and r radius having angle a in degrees
def getPointOnCircle(x, y, r, a):
  a = math.radians(a)
  cx = x + (r * math.cos(a))
  cy = y + (r * math.sin(a))
  return (cx[0], cy[0])

# number of candidate points
# can be changed
NC = 500

# to return conjugate and transpose
def H(a):
  return a.conj().T

# anchor position
anchor = fileContent["posBS"]

# from paper
distMu = 0.26
distSigmaSquare = 0.054 ** 2
# taking only 4 walls for MPCs
K = 4


# mu, sigma = 0.5, 0.1

# hard coded virtual angles
virtual_anchors = [(anchor[0][0], -1 * anchor[1][0]), (-1 * anchor[0][0], anchor[1][0]), 
                   (anchor[0][0], 12-anchor[1][0]), (11-anchor[0][0],anchor[1][0])]


# speed of light
C = 3 * (10 ** 8)


# plotting floor plan and a example tag and anchor position
# along with candidate points

floorP = []
for v in fileContent["floorplan"]:
  floorP.append((v[0],v[1]))
  floorP.append((v[2],v[3]))
tagIdx = 34
tag = fileContent["posEP"][34]

fig, ax = plt.subplots()
floorWalls = Path(floorP)
patch = patches.PathPatch(floorWalls, facecolor='orange', label='floor area')
ax.add_patch(patch)
ax.set_xlim(-2, 10)
ax.set_ylim(-2, 10)
ax.plot(anchor[0],anchor[1],"ko", label='anchor') # anchor 
ax.plot(tag[0],tag[1], "bo", label='tag')

adx = 0
mdx = 0


cirValue = fileContent["cir"][0][0][0][tagIdx]


dtwr = fileContent["distance"][adx][12][tagIdx]
# print(dtwr)
dnot = dtwr - distMu


radius = np.random.normal(dnot, distSigmaSquare, NC)
angle = np.random.uniform(0.0, 360.0, NC)

final_pos = None
min_cir = None
f1 = f2 = False
for candidateIdx, estimated_distance in enumerate(radius):
  estimated_angle = angle[candidateIdx]
  estimated_pos = getPointOnCircle(anchor[0], anchor[1], estimated_distance, estimated_angle)
  
  if floorWalls.contains_point((estimated_pos[0],estimated_pos[1])) == True:
    if not f1:
      ax.plot(estimated_pos[0],estimated_pos[1], 'go', label='candidate')
      f1 = True
    else:
      ax.plot(estimated_pos[0],estimated_pos[1], 'go')
  else:
    if not f2:
      ax.plot(estimated_pos[0],estimated_pos[1], 'ro', label='discarded')
      f2 = True
    else:
      ax.plot(estimated_pos[0],estimated_pos[1], 'ro')

ax.legend()
plt.title("Floor plan, anchor, tag and canditate points")
ax.set_xlabel("meters")
ax.set_ylabel("meters")
# ax.

ax.plot()
plt.show()

# algorithm taken from paper
position_error = []
final_pos_tag = []

for sdx, samples in enumerate(fileContent["cir"]):
  for adx, antenna in enumerate(samples):
    for mdx, measurements in enumerate(antenna):
      for tagIdx, cirValue in enumerate(measurements):

        # print(tagIdx,":", cirValue)
        dtwr = fileContent["distance"][adx][mdx][tagIdx]
        dnot = dtwr - distMu

        radius = np.random.normal(dnot, distSigmaSquare, NC)
        angle = np.random.uniform(0.0, 360.0, NC)
        
        final_pos = None
        min_cir = None
        for candidateIdx, estimated_distance in enumerate(radius):
          
          estimated_angle = angle[candidateIdx]
          estimated_pos = getPointOnCircle(anchor[0], anchor[1], estimated_distance, estimated_angle)
          # if patch.contains_point((estimated_pos[0],estimated_pos[1])) == False:
          #   continue
          estimated_cir = cirValue

          for vaIdx in range(K):
            distance_with_VA = math.hypot(virtual_anchors[vaIdx][0] - estimated_pos[0],
                                          virtual_anchors[vaIdx][1]- estimated_pos[1])
            estimated_delay = distance_with_VA / C

            # print(estimated_delay)
            esti_signal = getSDW(estimated_delay)

            estimated_alpha = H(esti_signal) * estimated_cir

            estimated_cir -= (estimated_alpha * esti_signal)
          
          if min_cir is None or min_cir < abs(cirValue - estimated_cir):
            min_cir = abs(cirValue - estimated_cir)
            final_pos = estimated_pos
        
        # here final position is estimated final position of the tag
        final_pos_tag.append((tagIdx, final_pos))

        tag_pos = fileContent["posEP"][tagIdx]
        pos_err = math.hypot(tag_pos[0]- final_pos[0], tag_pos[1] - final_pos[1])
        position_error.append(round(pos_err,3))

# may print very large output
pp.pprint(position_error)

# to save error to file
with open("/result_err.txt", 'w') as f:
  f.write(str(position_error))


with open("result_pos.txt", 'w') as f:
  f.write(str(final_pos_tag))

# to plot cdf of error in estimated positons

data = []
data = position_error
data = np.sort(data)
p = 1. * np.arange(len(data)) / (len(data) - 1)

fig, ax = plt.subplots()
ax.plot(data,p)
plt.title("NC = 50")
ax.set_xlabel('Error in meters')
ax.set_ylabel('probability')
plt.plot()
