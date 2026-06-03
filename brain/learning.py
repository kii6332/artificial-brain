 # 
 # This file is part of the artificial-brain distribution
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
import nn
import numpy as np
class learn:
	def __init__(self,weight,bias,active_function,learning_algrithm):
		self.weight=weight
		self.bias=bias
		self.active_function=active_function
		self.learning_algrithm=learning_algrithm
		if len(self.bias) != len(self.weight):
			raise ValueError(f"size of weight != size of bias : {len(self.bias)} != {len(self.weight)} | it have to be same\n more info:\n weight:{self.weight}\n bias:{self.bias}")
		self.na=nn.neural(self.active_function)
	def process(self,inp):
		return self.na.neural_network(inp,self.weight,self.bias)
	def error_calculation(self,inp,target):
		register_output=[]
		for t,p in zip(target,self.process(inp)):
			register_output.append(t-p)
		return register_output
	def tune(self,inp,target):
		output=self.process(inp)
		error=self.error_calculation(inp,target)
		return self.learning_algrithm(weight=self.weight,bias=self.bias,error=error,output=output)
	def evo(self,inp,target):
		apply_value = self.tune(inp,target)
		self.weight = apply_value["weight"]
		self.bias = apply_value["bias"]
#print(type(f))
a=[0,2,3,4,5]
w=[
[[2,3,4,5,6],[6,3,3,2,1]],
[[1,3],[4,5]],
[[1,3],[4,5],[1,2]]
]
b=[[10,15],[5,10],[2,1,100]]
def ga(error,weight,bias,output):
	return {"weight": weight, "bias": bias}
lr=learn(w,b,lambda x:x,ga)
print(lr.weight)
lr.evo(a,[950, 1800, 500])
print(lr.weight)
