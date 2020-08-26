import cv2
import os
from matplotlib import pyplot as plt
from math import ceil
import numpy as np
from itertools import product
inf = np.float('inf')



def helper(block1, block2, size, overlap):
	arr = ((block1[:, -overlap:] - block2[:, :overlap])**2).mean(2)
	minIndex = []
	lst = [list(arr[0])]
	for i in range(1, arr.shape[0]):
		l = [inf] + lst[-1] + [inf]
		l = np.array([l[:-2], l[1:-1], l[2:]])
		minArr = l.min(0)
		minArg = l.argmin(0) - 1
		minIndex.append(minArg)
		lstij = arr[i] + minArr
		lst.append(list(lstij))

	path = []
	minArg = np.argmin(lst[-1])
	path.append(minArg)

	for idx in minIndex[::-1]:
		minArg = minArg + idx[minArg]
		path.append(minArg)
	path = path[::-1]
	mask = np.zeros((size, size, block1.shape[2]))
	for i in range(len(path)):
		mask[i, :path[i]+1] = 1
	return mask

def findPatch(currBlock, currBlock1,texture, size, overlap, tolerance, direction):
	rows, columns = texture.shape[:2]
	arr = np.zeros((rows-size, columns-size)) + inf
	for i, j in product(range(rows-size), range(columns-size)):
		if direction=='hz':
			rmsVal = np.mean((texture[i:i+size, j:j+overlap] - currBlock[:, -overlap:])**2)
		else:
			if direction=='vt':
				rmsVal = np.mean((texture[i:i+overlap, j:j+size] - currBlock1[-overlap:, :])**2)
			else:
				rmsVal = np.mean((texture[i:i+size, j:j+overlap] - currBlock[:, -overlap:])**2)
				rmsVal += np.mean((texture[i:i+overlap, j:j+size] - currBlock1[-overlap:, :])**2)				
		if rmsVal > 0:
			arr[i, j] = rmsVal

	minVal = np.min(arr)
	y, x = np.where(arr < (1.0 + tolerance)*(minVal))
	c = np.random.randint(len(y))
	y, x = y[c], x[c]
	return texture[y:y+size, x:x+size]

def MinCutPatch(left_side, upper_side, curr_block, size, overlap,direction):
	mask1=None
	mask2=None
	if direction=='hz':
		mask1=helper(left_side, curr_block, size, overlap)
		return_Image = np.zeros(left_side.shape)
		return_Image[:, :overlap] = left_side[:, -overlap:]
		return_Image = return_Image*mask1 + curr_block*(1-mask1)
		return return_Image
	else:
		if direction=='vt':
			mask2=helper(np.rot90(upper_side), np.rot90(curr_block), size, overlap)
			return_Image = np.zeros(upper_side.shape)
			return_Image[:, :overlap] = upper_side[:, -overlap:]
			return_Image = np.rot90(return_Image*mask2 + curr_block*(1-mask2),3)
			return return_Image
		else:
			mask1=helper(left_side, curr_block, size, overlap)
			mask2=np.rot90(helper(np.rot90(upper_side), np.rot90(curr_block), size, overlap),3)
	
	mask2[:overlap, :overlap] = np.maximum(mask2[:overlap, :overlap] - mask1[:overlap, :overlap], 0)

	return_Image = np.zeros(curr_block.shape)
	return_Image[:, :overlap] = mask1[:, :overlap]*left_side[:, -overlap:]
	return_Image[:overlap, :] = return_Image[:overlap, :] + mask2[:overlap, :]*upper_side[-overlap:, :]
	return_Image = return_Image + (1-np.maximum(mask1, mask2))*curr_block
	return return_Image

def TextureMap(image, size, overlap, out_rows, out_columns, tolerance):
	new_rows = int(ceil((out_rows - size)*1.0/(size - overlap)))
	new_columns = int(ceil((out_columns - size)*1.0/(size - overlap)))
	textureMap = np.zeros(((size + new_rows*(size - overlap)), (size + new_columns*(size - overlap)), image.shape[2]))
	rows, columns = image.shape[:2]
	randH = np.random.randint(rows - size)
	randW = np.random.randint(columns - size)
	startBlock = image[randH:randH+size, randW:randW+size]
	textureMap[:size, :size, :] = startBlock

	for i, blkIdx in enumerate(range((size-overlap), textureMap.shape[1]-overlap, (size-overlap))):
		currBlock = textureMap[:size, (blkIdx-size+overlap):(blkIdx+overlap)]
		curr_block = findPatch(currBlock,None, image, size, overlap, tolerance,'hz')
		minCutPatch = MinCutPatch(currBlock,None, curr_block, size, overlap,'hz')
		textureMap[:size, (blkIdx):(blkIdx+size)] = minCutPatch

	for i, blkIdx in enumerate(range((size-overlap), textureMap.shape[0]-overlap, (size-overlap))):
		currBlock = textureMap[(blkIdx-size+overlap):(blkIdx+overlap), :size]
		curr_block = findPatch(None,currBlock, image, size, overlap, tolerance,'vt')
		minCutPatch = MinCutPatch(None,currBlock, curr_block, size, overlap,'vt')
		textureMap[(blkIdx):(blkIdx+size), :size] = minCutPatch
		
	for i in range(1, new_rows+1):
		for j in range(1, new_columns+1):
			index_i = i*(size-overlap)
			index_j = j*(size-overlap)
			left_side = textureMap[(index_i):(index_i+size), (index_j-size+overlap):(index_j+overlap)]
			upper_side  = textureMap[(index_i-size+overlap):(index_i+overlap), (index_j):(index_j+size)]
			curr_block = findPatch(left_side, upper_side, image, size, overlap, tolerance,'both')
			minCutPatch = MinCutPatch(left_side, upper_side, curr_block, size, overlap,'both') 
			textureMap[(index_i):(index_i+size), (index_j):(index_j+size)] = minCutPatch
	return textureMap

if __name__ == "__main__":
	path = "./textures/t19.png"
	size = 60
	scale = 2
	overlap = int(size/6)
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255.0
	plt.imshow(image)
	plt.show()
	input_rows, input_columns = image.shape[:2]
	rows, columns = int(scale*input_rows), int(scale*input_columns)
	textureMap = TextureMap(image, size, overlap, rows, columns, 0.1)
	plt.imshow(textureMap)
	plt.show()

