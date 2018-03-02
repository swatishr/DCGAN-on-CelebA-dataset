#Library file having functions for:

#1. Download the CelebA dataset
#2. Generate batch of image data that we will feed to the GAN
#3. Crop the images to get more focus on the face
#4. Normalize the images so that the pixel values are in the range of -0.5 to 0.5
#5. Downscale all the images to 28x28 (Image quality is lost, but training speeds up)
#6. to plot images in a square grid

import os
import zipfile
import requests
import math
from tqdm import tqdm 
from PIL import Image
from matplotlib import pyplot
import numpy as np

#get batch of image data in ndarray
def get_batch(images, width, height, mode="RGB"):
	#convert the batch of images obtained into ndarray
	data = np.array([get_image(single_img, width, height, mode) for single_img in images]).astype(np.float32)
	#print(np.shape(data))
	#print(len(data.shape))
	if(len(data.shape) < 4):
		data = data.reshape(data.shape + (1,))
	#print(len(data.shape))
	return data

#Read image
def get_image(imgpath, width, height, mode):
	image = Image.open(imgpath)
	if(image.size != (width,height)):
		face_width = face_height = 108
		width_offset = (image.size[0] - face_width) //2
		height_offset = (image.size[1] - face_height) // 2
		image = image.crop([width_offset, height_offset, width_offset + face_width, height_offset + face_height])
		image = image.resize([width, height], Image.ANTIALIAS)
	return np.array(image.convert(mode))

#Generate batches of data
def get_batches(batch_size, shape, data_files):
	IMAGE_MAX_VALUE = 255
	current_index = 0

	while (current_index+batch_size < shape[0]):
		data_batch = get_batch(data_files[current_index:current_index+batch_size], *shape[1:3]) #second parameter is to send variable number of arguments
		current_index+=batch_size
		#Normalization: to have the pixel values in the range of -0.5 to 0.5 and yield returns the value to the caller 
		#keeping the local variables of the function as it is in the memory and continuing the execution 
		#from where it left off
		yield data_batch/IMAGE_MAX_VALUE - 0.5

def download_celeb_a():
	#directory to store the CelebA dataset
	dirpath = './data'
	data_dir = 'celebA'

	#if the data already exists, then just ignore and return
	if(os.path.exists(os.path.join(dirpath, data_dir))):
		print("CelebA data already downloaded.")
		return

	#else we need to download the zip from Google drive and save it at location dirpath with name as filename
	filename, drive_id = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
	save_path = os.path.join(dirpath, filename)

	if(os.path.exists(save_path)):
		print("The Celeb zip folder already exists")
	else:
		download_from_google_drive(drive_id, save_path)

	#extract the downloaded zip file, remove the zip file and rename the extracted fodler as celebA (data_dir)
	zip_dir=""
	with zipfile.ZipFile(save_path) as zf:
		zip_dir = zf.namelist()[0]
		print("Extracting...")
		zf.extractall(dirpath)

	os.remove(save_path)
	os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))


def download_from_google_drive(driveid, destpath):
	print("Downloading CelebA data...")
	URL = "https://docs.google.com/uc?export=download"
	session = requests.Session()

	response = session.get(URL, params={'id': driveid}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': driveid, 'confirm':token}
		response = session.get(URL, params=params, stream=True)

	save_response(response,destpath)
	print("Download done!")

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith("download_warning"):
			return value
	return None

def save_response(response, destpath, chunk_size=32*1024):
	totalsize = int(response.headers.get('content-length', 0))
	with open(destpath, "wb") as f:
		for chunk in tqdm(response.iter_content(chunk_size), total=totalsize, unit = 'B', unit_scale=True, desc=destpath):
			if chunk:
				f.write(chunk)

#Function to save images as square grid
def images_square_grid(images, mode='RGB'):
	#get max size for square grid of images
	save_size = math.floor(np.sqrt(images.shape[0]))

	#scale to 0-255
	images = (((images - images.min()) * 255) / (images.max() - images.min())).astype(np.uint8)

	# Put images in a square arrangement
	images_in_square = np.reshape(images[:save_size*save_size],(save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))

	# Combine images to grid image
	new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
	for col_i, col_images in enumerate(images_in_square):
		for image_i, image in enumerate(col_images):
			im = Image.fromarray(image, mode)
			new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

	return new_im