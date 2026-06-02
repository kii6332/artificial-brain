 # 
 # This file is part of the artifical-brain distribution
 # Copyright (c) 2026 kii_6332.
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #
import numpy as np
from fractions import Fraction
import random as rand
class neural:
	def __init__(self,active_function= lambda x: 1 / (1+np.exp(-x)) ):
		self.active_function=active_function
	#simple active node calculation
	#sum(input*weight)-bias
	def active_node(self,inp,weight,bias=0):
		weight_array=[]
		if len(weight) != len(inp):
			raise ValueError(f"size of input != size of weight: {len(weight)} != {len(inp)} | it have to be same\n more info:\n input:{inp}\n weight:{weight}")
		for i,w in zip(inp,weight):
			weight_array.append(i*w)
		at= sum(weight_array)-bias
		return self.active_function(at)
	# :O many nodes
	#just active node but for many output node in current layer
	def multi_active_node(self,inp,weight,bias):
		outcome=[]
		if len(bias) != len(weight):
			raise ValueError(f"size of weight != size of bias : {len(bias)} != {len(weight)} | it have to be same\n more info:\n weight:{weight}\n bias:{bias}")
		for w,b in zip(weight,bias):
			outcome.append(self.active_node(inp,w,b))
		return outcome
	#wow is that da neural network !!??
	#just muti active node with many layer
	def neural_network(self,inp,weight,bias):
		register_input=inp
		for w,b in zip(weight,bias):
			register_input=self.multi_active_node(register_input,w,b)
		return register_input

#test stuff down here
if __name__ == "__main__":
	function_check=0
	fail_function=[]
	error_check=0
	error_fail=[]
	nq=neural(active_function=lambda x:x)
	a=[0,2,3,4,5]
	w=[
	[[2,3,4,5,6],[6,3,3,2,1]],
	[[1,3],[4,5]],
	[[1,3],[4,5],[1,2]]
	]
	b=[[10,15],[5,10],[2,1,100]]
	print("function test 1/3 \n")
	try:
		ot = nq.active_node(a,[2,3,4,5,6],3)
		if ot == 65:
			function_check +=1
	except Exception:
		fail_function.append("active_node")
	print("function test 2/3 \n")
	try:
		ot=nq.multi_active_node(a,[[2,3,4,5,6],[6,3,3,2,1]],[1,3])
		if ot == [67, 25]:
			function_check +=1
	except Exception:
		fail_function.append("multi_active_node")
	print("function test 3/3 \n")
	try:
		ot = nq.neural_network(a,w,b)
		if ot == [951, 1802, 566]:
			function_check +=1
	except Exception:
		fail_function.append("neural_network")
	w=[
	[[2,3,4,5,6],[6,3,3,2,1]],
	[[1,3],[4,5]],
	[[1,3],[4,5],[1,2,1]]
	]
	#error checking
	print("error test 1/2 \n")
	try:
		ot=nq.neural_network(a,w,b)
		error_fail.append("weight != input")
	except:
		error_check+=1
	w=[
	[[2,3,4,5,6],[6,3,3,2,1]],
	[[1,3],[4,5]],
	[[1,3],[4,5],[1,2]]
	]
	b=[[10,15],[5,10],[2,1,100,10]]
	print("error test 2/2 \n")
	try:
		nq.neural_network(a,w,b)
		error_fail.append("bias != weight")
	except:
		error_check+=1
	print(f"summary:\n function test pass: {function_check}/3\n fail function: {fail_function}\nerror test pass: {error_check}/2\n fail error: {error_fail}")
