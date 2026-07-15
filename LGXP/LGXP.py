
import numpy as np
# from LGXP import gabor
# from LGXP import LGP

import cv2

# encoding: utf-8

import numpy as np
import cv2 as cv

src_f = 0
kernel_size =21
pos_sigma = 5
pos_lm = kernel_size-2
pos_th = 0
pos_gam = 100
pos_psi = 90

def Process():
	sig = pos_sigma
	lm = pos_lm+2
	th = pos_th*np.pi/180.
	gm = pos_gam/100.
	ps = (pos_psi-180)*np.pi/180
	print('kern_size=' + str(kernel_size) + ', sig=' + str(sig) + ', th=' + str(th) + ', lm=' + str(lm) +', gm=' + str(gm) + ', ps=' + str(ps))
	kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
	kernelimg = kernel/2.+0.5
	global src_f
	dest = cv.filter2D(src_f, cv.CV_32F,kernel)
	cv.imshow('Process window', dest)
	cv.imshow('Kernel', cv.resize(kernelimg, (kernel_size*20,kernel_size*20)))
	cv.imshow('Mag', np.power(dest,2))

def gabor(img):
	sig = pos_sigma
	lm = pos_lm+2
	th = pos_th*np.pi/180.
	gm = pos_gam/100.
	ps = (pos_psi-180)*np.pi/180
	kernel = cv.getGaborKernel((kernel_size,kernel_size),sig,th,lm,gm,ps)
	kernelimg = kernel/2.+0.5
	dest = cv.filter2D(img, cv.CV_32F,kernel)
	# cv.imshow('Process window', dest)
	# cv.imshow('Kernel', cv.resize(kernelimg, (kernel_size*20,kernel_size*20)))
	# cv.imshow('Mag', np.power(dest,2))
	return dest

def cb_sigma(pos):
	global pos_sigma
	if pos > 0:
		pos_sigma = pos
	else:
		pos_sigma = 1
	Process()

def cb_lm(pos):
	global pos_lm
	pos_lm = pos
	Process()

def cb_th(pos):
	global pos_th
	pos_th = pos
	Process()

def cb_psi(pos):
	global pos_psi
	pos_psi = pos
	Process()

def cb_gam(pos):
	global pos_gam
	pos_gam = pos
	Process()

# if __name__ == '__main__':
#
# 	image = cv.imread('bolin.jpg',1)
# 	cv.imshow('Src',image)
# 	src = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
# 	#global src_f
# 	src_f = np.array(src, dtype=np.float32)
# 	src_f /= 255.
# 	if not kernel_size%2:
# 		kernel_size += 1
#
# 	cv.namedWindow('Process window',1)
# 	cv.createTrackbar('Sigma','Process window',pos_sigma,int(kernel_size/2),cb_sigma)
# 	cv.createTrackbar('Lambda', 'Process window', pos_lm, kernel_size-2, cb_lm)
# 	cv.createTrackbar('Theta', 'Process window', pos_th, 360, cb_th)
# 	cv.createTrackbar('gamma', 'Process window', pos_psi, 300, cb_gam)
# 	cv.createTrackbar('Psi', 'Process window', pos_psi, 360, cb_psi)
# 	Process()
# 	cv.waitKey(0)
# 	cv.destroyAllWindows()

def LGP(img, r=1):
    padded = np.pad(img, (r, r), 'constant')
    a1 = padded[:-2*r, :-2*r]
    b1 = padded[:-2*r, r:-r]
    a2 = padded[:-2*r, 2*r:]
    b2 = padded[r:-r, 2*r:]
    a3 = padded[2*r:, 2*r:]
    b3 = padded[2*r:, r:-r]
    a4 = padded[2*r:, :-2*r]
    b4 = padded[r:-r, :-2*r]
    codes = (a1 >= a3) + 2*(a2 >= a4) + 4*(b1 >= b3) + 8*(b2 >= b4)
    return codes

def get_LGXP(img):
    gabor_1=gabor(img)
    gabor_1=cv2.normalize(src=gabor_1, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    lgp_1=LGP(img)
    lgp_1=lgp_1.astype(np.float)
    out=lgp_1*0.7+gabor_1*0.3
    out = cv2.normalize(src=out, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    out =np.histogram(out,bins=100)[0]
    # out=np.mean(out,axis=0)
    return out