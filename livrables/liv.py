import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import revert
import math
import matplotlib.pyplot as plt
import os

def cross_correlation (ya, yb):
    """ Cross-correlation of N_batch x N tensors. """
    ya, yb = ya - ya.mean([0]), yb - yb.mean([0])
    yab = ya.T @ yb
    return yab / (ya.norm(dim=0)[:,None] * yb.norm(dim=0))

class encoderICP(nn.Module):
	#dim_input = [n_channels, n_points]
	#dim_output is an integer
	def __init__(self, dim_input, dim_output):
		super().__init__()
		self.dim_input = dim_input
		self.dim_output = dim_output
		k = 5
		self.l1 = nn.Sequential(nn.Conv1d(1, 4, k, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l2 = nn.Sequential(nn.Conv1d(4, 16, k, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l3 = nn.Sequential(nn.Conv1d(16, 64, k, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l4 = nn.Sequential(nn.Flatten(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128,dim_output))

	def forward(self,x):
		x = self.l1(x)
		x = nn.functional.interpolate(x,size=32)
		x = self.l2(x)
		x = nn.functional.interpolate(x,size=8)
		x = self.l3(x)
		x = nn.functional.interpolate(x,size=2)
		x = self.l4(x)
		return x
		
class encoderMRI(nn.Module):
	#dim_input = [n_channels, n_points]
	#dim_output is an integer
	def __init__(self, dim_input, dim_output):
		super().__init__()
		self.dim_input = dim_input
		self.dim_output = dim_output
		k = 5
		self.l1 = nn.Sequential(nn.Conv1d(6, 24, k, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l2 = nn.Sequential(nn.Conv1d(24, 48, k, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l3 = nn.Sequential(nn.Conv1d(48, 96, k-2, padding_mode='circular'), nn.ReLU())#.to(device)
		self.l4 = nn.Sequential(nn.Flatten(), nn.Linear(192, 192), nn.ReLU(), nn.Linear(192,dim_output))

	def forward(self,x):
		x/=100
		x = self.l1(x)
		x = nn.functional.interpolate(x,size=8)
		x = self.l2(x)
		x = nn.functional.interpolate(x,size=4)
		x = self.l3(x)
		x = nn.functional.interpolate(x,size=2)
		x = self.l4(x)
		return x
		
class Transformation(nn.Module):
	def __init__(self, classes):
		super().__init__()
		x = torch.rand(1)
		n = len(classes)
		self.tr = classes[(x*n).int().item()]()
		
	def forward(self, x):
		return self.tr(x)
		
class Noise(nn.Module):
	def __init__(self, I=0.1):
		super().__init__()
		self.I = I
	
	def forward(self,x):
		noise = torch.randn(x.shape)*self.I
		return x + noise
		
class BarlowTwins():
	def __init__(self, model_class, dim_input, dim_output):
		self.model = model_class(dim_input, dim_output)
		
	def train(self, opt, dataloader, verbose = False):
		for i,x in enumerate(dataloader):
			x = x.view(-1,self.model.dim_input[0],self.model.dim_input[1])
			#in facts, the argument above will be (-1,1,128) for icp and (-1,6,32) for mri
			tr1, tr2 = Transformation([Noise]), Transformation([Noise])
			model_input = torch.cat([tr1(x), tr2(x)], dim=0)
			model_output = self.model(model_input)
			y1, y2 = model_output[:len(model_output)//2], model_output[len(model_output)//2:]
			cr_cor = cross_correlation(y1,y2)
			loss = ((cr_cor - torch.eye(len(cr_cor)))**2).sum()
			opt.zero_grad()
			loss.backward()
			opt.step()
			if verbose and i%100 == 0:
				print("loss", loss.item())

class PulsesDataset(Dataset):
	def __init__(self, db):
		self.db = db
		self.pulses = db.view(-1, 128)
		
	def __len__(self):
		return len(self.pulses)
		
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.pulses[idx]
		
class FlowsDataset(Dataset):
	def __init__(self, db):
		self.db = db
		self.flows = db.view(-1, 6*32)
		
	def __len__(self):
		return len(self.flows)
		
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.flows[idx]
		
class HeadDataset(Dataset):
	def __init__(self, latent_vectors, labels):
		if(len(latent_vectors)!=len(labels)):
			raise Exception("HeadDataset : input data vector and label vector have different lengths")
		self.latent_vectors = latent_vectors
		self.labels = labels.float().view(-1,1)
	def __len__(self):
		return len(self.latent_vectors)
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return (self.latent_vectors[idx],self.labels[idx])

#return a BarlowTwins architecture containing an encoder that outputs vectors of size dim_output and supposed to take ICP pulses in input
def makeICP_BT(dim_output):
	return BarlowTwins(encoderICP, [1,128], dim_output)
	
#bt = barlowTwins (class defined above, see examples in mainICP() below)
#pulses has type torch.tensor
def trainICP_BT(bt, pulses, n_epochs = 5):
	ds = PulsesDataset(pulses)
	dataloader = DataLoader(ds, batch_size = 64, shuffle = True)
	opt = torch.optim.Adam(bt.model.parameters(), lr = 1e-3)
	for i in range(n_epochs):
		bt.train(opt, dataloader, verbose = True)

#returns a dictionnary containing latent_vectors and keys of patients of the database db. The latent vectors are calculated using bt, being supposed already trained
#bt = trained barlowtwins
#db = database, see examples in mainICP() below
def getLatentVectorsICP(bt, db):
	centroids = []
	for i in range(len(db['pulses'])):
		points = bt.model(db['pulses'][i].view(-1,1,128))
		centroid = points.mean(dim=0)
		centroids.append(centroid)
	centroids = torch.stack(centroids).detach()
	return {'latent_vectors':centroids, 'keys':db['keys']}

#generate random vector of length n containing 0s and 1s
def genLabels(n):
	labels = torch.randn(n)
	labels/=labels.abs()
	return ((labels+1)/2).int()

#return a BarlowTwins architecture containing an encoder that outputs vectors of size dim_output and supposed to take MRI flows in input
def makeMRI_BT(dim_output):
	return BarlowTwins(encoderMRI, [6,32], dim_output)
	
#bt = BarlowTwins (class defined above, see examples in mainMRI() below)
#flows has type torch.Tensor
def trainMRI_BT(bt, flows, n_epochs = 5):
	ds = FlowsDataset(flows)
	dataloader = DataLoader(ds, batch_size = 64, shuffle = True)
	opt = torch.optim.Adam(bt.model.parameters(), lr = 1e-3)
	for i in range(n_epochs):
		bt.train(opt, dataloader, verbose = True)

#returns a dictionnary containing latent_vectors and keys of patients of the database db. The latent vectors are calculated using bt, being supposed already trained
#bt = trained barlowtwins
#db = database, see examples in mainMRI() below
def getLatentVectorsMRI(bt, db):
	latent_vectors = bt.model(db['flows']).detach()
	return {'latent_vectors':latent_vectors, 'keys':db['keys']}
	
#generate head with wanted dimensions
def makeHead(dim_input, dim_output):
	dim_hidden = int(math.sqrt(dim_input*dim_output))
	return nn.Sequential(nn.Linear(dim_input,dim_hidden),
				nn.ReLU(),
				nn.Linear(dim_hidden,dim_hidden),
				nn.ReLU(),
				nn.Linear(dim_hidden,dim_output),
				nn.Sigmoid())

#train head		
def trainHead(head, latent_vectors, labels, n_epochs = 5, fn_loss = torch.nn.functional.binary_cross_entropy, verbose = True):
	ds = HeadDataset(latent_vectors, labels)
	dataloader = DataLoader(ds, batch_size = 64, shuffle = True)
	opt = torch.optim.Adam(head.parameters(), lr = 1e-3)
	for j in range(n_epochs):
		for i,z in enumerate(dataloader):
			x,y = z
			y_pred = head(x)
			loss = fn_loss(y_pred,y)
			opt.zero_grad()
			loss.backward()
			opt.step()
			if verbose and i%100 == 0:
				print("loss", loss.item())
				
def predictICP(bt, head, db):
	with torch.no_grad():
		lv = getLatentVectorsICP(bt, db)
		predicted_labels = head(lv['latent_vectors'])
	return {'predicted_labels':predicted_labels , 'keys':lv['keys']}
	
def predictMRI(bt, head, db):
	with torch.no_grad():
		lv = getLatentVectorsMRI(bt, db)
		predicted_labels = head(lv['latent_vectors'])
	return {'predicted_labels':predicted_labels , 'keys':lv['keys']}
	
def predictICP_MRI(btICP, btMRI, headICP_MRI, db):
	with torch.no_grad():
		lvICP = getLatentVectorsICP(btICP, db)
		lvMRI = getLatentVectorsMRI(btMRI, db)
		lv = torch.cat([lvICP['latent_vectors'],lvMRI['latent_vectors']], dim = 1)
		predicted_labels = headICP_MRI(lv)
	return {'predicted_labels':predicted_labels , 'keys':db['keys']}

def cartesianProduct(dbICP, dbMRI):
	pl,fl = dbICP['pulses'], dbMRI['flows']
	n,m = len(pl), len(fl)
	#à finir, faire des repeat sur tout (entre autre flows et pulses) mais un cat sur les keys
	keys = [("","")]*(n*m)
	for i in range(n):
		for j in range(m):
			keys[i*m + j] = (dbICP['keys'][i], dbMRI['keys'][j])
	return {'pulses': dbICP['pulses'].repeat(m,1,1), 'flows':dbMRI['flows'].repeat(1,n,1).view(-1, fl.shape[1], fl.shape[2]), 'keys' : keys}
	

def mainICP():
	if "INFUSION_DATASETS" in os.environ:
		dbpath = os.environ["INFUSION_DATASETS"]
		assert os.path.isdir(dbpath)
		dbpath = os.path.join(dbpath, "baseline-full.pt")#you might have to change this string if you don't use the same database name
		db = torch.load(dbpath)
		pulses = db['pulses']
		bt = makeICP_BT(16)
		trainICP_BT(bt,pulses, n_epochs = 5)
		lv = getLatentVectorsICP(bt, db)
		print(lv['latent_vectors'])
		h = makeHead(16,1)
		#random labels
		labels = genLabels(len(lv['latent_vectors']))
		trainHead(h, lv['latent_vectors'], labels, n_epochs = 20)
		print(predictICP(bt, h, db)['predicted_labels'])
	else:
		print("INFUSION_DATASETS environment variable not defined")

def mainMRI():
	if "PCMRI_DATASETS" in os.environ:
		dbpath = os.environ["PCMRI_DATASETS"]
		assert os.path.isdir(dbpath)
		dbpath = os.path.join(dbpath, "full.pt")#you might have to change this string if you don't use the same database name
		db = torch.load(dbpath)
		flows = db['flows']
		bt = makeMRI_BT(16)
		trainMRI_BT(bt, flows, n_epochs = 10)
		lv = getLatentVectorsMRI(bt, db)
		print(lv['latent_vectors'])
		h = makeHead(16,1)
		#random labels
		labels = genLabels(len(lv['latent_vectors']))
		trainHead(h, lv['latent_vectors'], labels, n_epochs = 20)
		print(predictMRI(bt, h, db)['predicted_labels'])
	else:
		print("PCMRI_DATASETS environment variable not defined")
	
def mainICP_MRI():
	if "PCMRI_DATASETS" in os.environ and "INFUSION_DATASETS" in os.environ :
		dbpathICP = os.environ["INFUSION_DATASETS"]
		assert os.path.isdir(dbpathICP)
		dbpathICP = os.path.join(dbpathICP, "baseline-full.pt")#you might have to change this string if you don't use the same database name
		dbpathMRI = os.environ["PCMRI_DATASETS"]
		assert os.path.isdir(dbpathMRI)
		dbpathMRI = os.path.join(dbpathMRI, "full.pt")
		dbICP = torch.load(dbpathICP)
		dbMRI = torch.load(dbpathMRI)
		#we restrict to the first 500 patients of each database and we consider that the i-th patients of the first and the second database are the same, this is just for testing the algorithm with joint latent vectors
		dbICP['pulses'] = dbICP['pulses'][:500]
		dbMRI['flows'] = dbMRI['flows'][:500]
		pulses = dbICP['pulses']
		flows = dbMRI['flows']
		btICP = makeICP_BT(16)
		btMRI = makeMRI_BT(16)
		trainICP_BT(btICP, pulses, n_epochs = 5)
		trainMRI_BT(btMRI, flows, n_epochs = 10)
		lvICP = getLatentVectorsICP(btICP, dbICP)
		lvMRI = getLatentVectorsMRI(btMRI, dbMRI)
		lv = torch.cat([lvICP['latent_vectors'],lvMRI['latent_vectors']], dim = 1)
		h = makeHead(16+16,1)
		labels = genLabels(len(lv))
		trainHead(h, lv, labels, n_epochs = 5)
		db = {'pulses' : dbICP['pulses'], 'flows' : dbMRI['flows'], 'keys' : dbICP['keys']}
		print(predictICP_MRI(btICP, btMRI, h, db)['predicted_labels'])
	else:
		print("One of these environment variables is not defined : INFUSION_DATASETS, PCMRI_DATASETS ")

