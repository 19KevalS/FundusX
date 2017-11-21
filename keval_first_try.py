# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
from PIL import Image

import argparse
import sys
import os
import csv
import numpy 



from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def getLabels (string):
	trainlabels = []
	updatedlabels = []
	
	with open(string, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			x = int(row[1])
			trainlabels.append(x)
		

		for x in trainlabels:
			if x == 0:
				label = [1,0,0,0,0]
				updatedlabels.append(label)
			if x == 1:
				label = [0,1,0,0,0]
				updatedlabels.append(label)
			if x == 2:
				label = [0,0,1,0,0]
				updatedlabels.append(label)
			if x == 3:
				label = [0,0,0,1,0]
				updatedlabels.append(label)
			if x == 4:
				label = [0,0,0,0,1]
				updatedlabels.append(label)
		
		
				
	return (numpy.array(updatedlabels)) #This is a bad way of doing this. Takes up too much memory
		
def importData(): 
	train_images = []
	train_labels = getLabels('/Users/kevalshah/Desktop/trainLabels.csv')
	path = r'/Users/kevalshah/Desktop/train'
	    
	pathlist = Path(path).glob('**/*.jpeg')
	    
	for path in pathlist:
		jpeg = Image.open(path) #FIX it is not iterating through properly
		x = numpy.array(jpeg.getdata())
		x = x[:, 0]
		train_images.append(x)
                
                # train should be a matrix of size (a,b)
                # where a is the number of examples, b is the total of pixels

                # pending questions:
                # 1. revisit how to extract RGB into one value
                # 2. reduce the per image data 
                #    http://effbot.org/imagingbook/image.htm#tag-Image.Image.getpixel
        
		
	test_images = []
	test_labels = getLabels('/Users/kevalshah/Desktop/testLabels.csv')
	path = r'/Users/kevalshah/Desktop/test'
	
	pathlist2 = Path(path).glob('**/*.jpeg')
	
	for path in pathlist2:
		jpeg = Image.open(path)
		x = numpy.array(jpeg.getdata())
		x = x[:,0]
		test_images.append(x)
        
	

	return (numpy.array(train_images), train_labels, numpy.array(test_images), test_labels)

def main(_):
  # Import data
  	
	mnist = importData()

  # Create the model
	x = tf.placeholder(tf.float32, [None, 15054336])
	W = tf.Variable(tf.zeros([15054336, 5]))
	b = tf.Variable(tf.zeros([5]))
	y = tf.matmul(x, W) + b

  # Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 5])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
  # Train

	sess.run(train_step, feed_dict={x: mnist[0], y_: mnist[1]})

  # Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist[2], y_: mnist[3]}))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
