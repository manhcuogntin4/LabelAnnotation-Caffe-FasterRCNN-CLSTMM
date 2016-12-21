import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import sys, getopt
import os

CACHE_FOLDER = '/tmp/caffe_demos_uploads/cache'
this_dir = os.path.dirname(__file__)


def convert_to_binary(img):
	if (img.shape >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	res = cv2.fastNlMeansDenoising(res,20, 7, 21)
	out_path = os.path.join(CACHE_FOLDER, "out.png")
	cv2.imwrite(out_path,res)
	return out_path, res


def extract_text(img_path, model_path):
	ocr = pyclstm.ClstmOcr()
	ocr.load(model_path)
	imgFile = Image.open(img_path)
	text = ocr.recognize(imgFile)
	text.encode('utf-8')
	chars = ocr.recognize_chars(imgFile)
	prob = 1
	index = 0
	# print text
	if(text.find(u':') != -1 and text.index(u':') < 3):
		index = text.index(u':')
	if(text.find(u' ') != -1 and (text.index(u' ') <= 5)):
		index = text.index(u' ')
	for ind, j in enumerate(chars):
		if ind >= index:		
			prob *= j.confidence
	return text, prob


def crop_image(img,cropX=0, cropY=0, cropWidth=0, cropHeight=0):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	res=img[cropY:h-cropHeight, cropX:w-cropWidth]
	out_path = os.path.join(CACHE_FOLDER, "croped.png")
	cv2.imwrite(out_path,res)
	return out_path


def clstm_ocr(img, islieu=False):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	model_path = os.path.join(this_dir, 'model-nomprenom2911-binary.clstm')
	if islieu:
		model_path = os.path.join(this_dir, 'model-lieu2911-binary.clstm')
	converted_image_path, image = convert_to_binary(img)
	maxPro = 0
	ocr_result = ""
	for i in range(0, 3):
		for j in range(0, 3):
			for k in range(0, 3):
				for h in range(0, 3):
					img_path = crop_image(image, i, j, k, h)
					text, prob = extract_text(img_path, model_path)
					if(prob > maxPro):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2):
						break
	return (ocr_result, maxPro)


if __name__ == '__main__':
	filename = os.path.join(this_dir, 'demo', 'prenom0.png')
	img = cv2.imread(filename)
	print clstm_ocr(img)