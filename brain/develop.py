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
import learning
import numpy as np
import random as rand
def rain(error,weight,bias,output):
	weight=np.array(error) * np.array(weight)
	bias[-1]=bias[-1]-error
	#print(weight)
	return {"weight":weight,"bias":bias}
def summary(data):
	sm=[0,0,0]
	data=list(data)
	sm[data.index(max(data))]=1
	return sm
def re(data):
	return data[-1:]+data[:-1]
lr=learning.learn(weight=[
[[3,1,1],[1,3,1],[1,1,3]],
[[2,1,3],[3,2,1],[1,3,2]],
[[3,1,1],[1,3,1],[1,1,3]]
],
bias=[
[4,4,4],
[4,4,4],
[2,2,2]
],
learning_algrithm= rain)
#print(lr.process([1,0,0]))
#lr.evo([1,0,0],[0,1,0])
#print(lr.process([1,0,0]))
#lr.evo([0,1,0],[0,0,1])
#print(lr.process([0,1,0]))
#lr.evo([0,0,1],[0,1,0])
#print(lr.process([0,0,1]))
#print(lr.weight)
#print(lr.bias)
#print(lr.process([0.5,0.5,0.5]))
game= [0,1,0]
lan=["rock","paper","scissors"]
for i in range(100):
	rand.shuffle(game)
	ai=summary(lr.process([0.5,0.5,0.5]))
	ai_lang=lan[ai.index(max(ai))]
	ran_lang=lan[game.index(max(game))]
	print(f"AI:{ai_lang}		random:{ran_lang}	data:{lr.process([0.5,0.5,0.5])}")
	lr.evo([0.5,0.5,0.5],re(game))
