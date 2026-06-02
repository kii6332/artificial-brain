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
	def __init__(self,act_fun= lambda x: 1 / (1+np.exp(-x)) ):
		self.act_fun=act_fun
		
	def at_node(self,ip,we,ba=0):
		wa=[]
		if len(we) != len(ip):
			raise ValueError(f"size of input != size of weight: {len(we)} != {len(ip)} | it have to be same\n more info:\n input:{ip}\n weight:{we}")
		for i,w in zip(ip,we):
			wa.append(i*w)
		#print(wa)
		at= sum(wa)-ba
		#print(at)
		return self.act_fun(at)
	def multi_at_node(self,ip,we,ba):
		op=[]
		if len(ba) != len(we):
			raise ValueError(f"size of weight != size of bias : {len(ba)} != {len(we)} | it have to be same\n more info:\n weight:{we}\n bias:{ba}")
		for w,b in zip(we,ba):
			op.append(self.at_node(ip,w,b))
		return op
	def neural_network(self,ip,we,ba):
		p=ip
		for w,b in zip(we,ba):
			p=self.multi_at_node(p,w,b)
		return p

'''class neural_network:
	def __init__(self,inp,hide,out,act_fun= lambda x: 1 / (1+np.exp(-x))):
		self.inp=inp
		self.hide=hide
		self.out=out
		nq=neural(act_fun=act_fun)
		'''
		
nq=neural(act_fun=lambda x:x)
a=[0,2,3,4,5]
w=[
[[2,3,4,5,6],[6,3,3,2,1]],
[[1,3],[4,5]],
[[1,3],[4,5],[1,2]]
]
b=[[10,15],[5,10],[2,1,100]]
p=a
print(nq.neural_network(a,w,b))
