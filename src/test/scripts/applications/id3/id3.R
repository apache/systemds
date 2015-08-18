#-------------------------------------------------------------
#
# (C) Copyright IBM Corp. 2010, 2015
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
#
#-------------------------------------------------------------

args <- commandArgs(TRUE)

library("Matrix")
library("matrixStats")

options(warn=-1)


id3_learn = function(X, y, X_subset, attributes, minsplit)
{
	#try the two base cases first

	#of the remaining samples, compute a histogram for labels
	hist_labels_helper = as.matrix(aggregate(as.vector(X_subset), by=list(as.vector(y)), FUN=sum))
	hist_labels = as.matrix(hist_labels_helper[,2])

	#go through the histogram to compute the number of labels
	#with non-zero samples
	#and to pull out the most popular label
	
	num_non_zero_labels = sum(hist_labels > 0);
	most_popular_label = which.max(t(hist_labels));
	num_remaining_attrs = sum(attributes)
	
	num_samples = sum(X_subset)
	mpl = as.numeric(most_popular_label)
	
	nodes = matrix(0, 1, 1)
	edges = matrix(0, 1, 1)
	
	#if all samples have the same label then return a leaf node
	#if no attributes remain then return a leaf node with the most popular label	
	if(num_non_zero_labels == 1 | num_remaining_attrs == 0 | num_samples < minsplit){
		nodes = matrix(0, 1, 2)
		nodes[1,1] = -1
		nodes[1,2] = most_popular_label
		edges = matrix(-1, 1, 1)
	}else{
		#computing gains for all available attributes using parfor
		hist_labels2_helper = as.matrix(aggregate(as.vector(X_subset), by=list(as.vector(y)), FUN=sum))
		hist_labels2 = as.matrix(hist_labels2_helper[,2])
		num_samples2 = sum(X_subset)
		zero_entries_in_hist1 = (hist_labels2 == 0)
		pi1 = hist_labels2/num_samples2
		log_term1 = zero_entries_in_hist1*1 + (1-zero_entries_in_hist1)*pi1
		entropy_vector1 = -pi1*log(log_term1)
		ht = sum(entropy_vector1)
		
		sz = nrow(attributes)
		gains = matrix(0, sz, 1)
		for(i in 1:nrow(attributes)){
			if(as.numeric(attributes[i,1]) == 1){
				attr_vals = X[,i]
				attr_domain_helper = as.matrix(aggregate(as.vector(X_subset), by=list(as.vector(attr_vals)), FUN=sum))
				attr_domain = as.matrix(attr_domain_helper[,2])

				hxt_vector = matrix(0, nrow(attr_domain), 1)
				
				for(j in 1:nrow(attr_domain)){
					if(as.numeric(attr_domain[j,1]) != 0){
						val = j
						Tj = X_subset * (X[,i] == val)
						
						#entropy = compute_entropy(Tj, y)
						hist_labels1_helper = as.matrix(aggregate(as.vector(Tj), by=list(as.vector(y)), FUN=sum))
						hist_labels1 = as.matrix(hist_labels1_helper[,2])
						num_samples1 = sum(Tj)
						zero_entries_in_hist = (hist_labels1 == 0)
						pi = hist_labels1/num_samples1
						log_term = zero_entries_in_hist*1 + (1-zero_entries_in_hist)*pi
						entropy_vector = -pi*log(log_term)
						entropy = sum(entropy_vector)
	
						hxt_vector[j,1] = sum(Tj)/sum(X_subset)*entropy
					}
				}
				hxt = sum(hxt_vector)
				gains[i,1] = (ht - hxt)
			}
		}
		
		#pick out attr with highest gain
		best_attr = -1
		max_gain = 0
		for(i4 in 1:nrow(gains)){
			if(as.numeric(attributes[i4,1]) == 1){
				g = as.numeric(gains[i4,1])
				if(best_attr == -1 | max_gain <= g){
					max_gain = g
					best_attr = i4
				}
			}
		}		
		
		attr_vals = X[,best_attr]
		attr_domain_helper = as.matrix(aggregate(as.vector(X_subset), by=list(as.vector(attr_vals)), FUN=sum))
		attr_domain = as.matrix(attr_domain_helper[,2])

		new_attributes = attributes
		new_attributes[best_attr, 1] = 0
		
		max_sz_subtree = 2*sum(X_subset)
		sz2 = nrow(attr_domain)
		sz1 = sz2*max_sz_subtree
		
		tempNodeStore = matrix(0, 2, sz1)
		tempEdgeStore = matrix(0, 3, sz1)
		numSubtreeNodes = matrix(0, sz2, 1)
		numSubtreeEdges = matrix(0, sz2, 1)
		
		for(i1 in 1:nrow(attr_domain)){
			
			Ti = X_subset * (X[,best_attr] == i1)
			num_nodes_Ti = sum(Ti)
			
			if(num_nodes_Ti > 0){
				tmpRet <- id3_learn(X, y, Ti, new_attributes, minsplit)
			  nodesi = as.matrix(tmpRet$a);
        edgesi = as.matrix(tmpRet$b);
      
				start_pt = 1+(i1-1)*max_sz_subtree
        tempNodeStore[,start_pt:(start_pt+nrow(nodesi)-1)] = t(nodesi)
			
        numSubtreeNodes[i1,1] = nrow(nodesi)
				if(nrow(edgesi)!=1 | ncol(edgesi)!=1 | as.numeric(edgesi[1,1])!=-1){
					tempEdgeStore[,start_pt:(start_pt+nrow(edgesi)-1)] = t(edgesi)
					numSubtreeEdges[i1,1] = nrow(edgesi)
				}else{
					numSubtreeEdges[i1,1] = 0
				}
			}
		}
		
		num_nodes_in_subtrees = sum(numSubtreeNodes)
		num_edges_in_subtrees = sum(numSubtreeEdges)
		
		#creating root node
		sz = 1+num_nodes_in_subtrees
		
		nodes = matrix(0, sz, 2)
		nodes[1,1] = best_attr
		numNodes = 1
		
		#edges from root to children
		sz = sum(numSubtreeNodes > 0) + num_edges_in_subtrees
		
		edges = matrix(1, sz, 3)
		numEdges = 0
		for(i6 in 1:nrow(attr_domain)){
			num_nodesi = as.numeric(numSubtreeNodes[i6,1])
			if(num_nodesi > 0){
				edges[numEdges+1,2] = i6
				numEdges = numEdges + 1
			}
		}
		
		nonEmptyAttri = 0
		for(i7 in 1:nrow(attr_domain)){
			numNodesInSubtree = as.numeric(numSubtreeNodes[i7,1])
		
			if(numNodesInSubtree > 0){
				start_pt1 = 1 + (i7-1)*max_sz_subtree
				nodes[(numNodes+1):(numNodes+numNodesInSubtree),] = t(tempNodeStore[,start_pt1:(start_pt1+numNodesInSubtree-1)])
			
				numEdgesInSubtree = as.numeric(numSubtreeEdges[i7,1])
			
				if(numEdgesInSubtree!=0){
					edgesi1 = t(tempEdgeStore[,start_pt1:(start_pt1+numEdgesInSubtree-1)])
					edgesi1[,1] = edgesi1[,1] + numNodes
					edgesi1[,3] = edgesi1[,3] + numNodes
          
					edges[(numEdges+1):(numEdges+numEdgesInSubtree),] = edgesi1
					numEdges = numEdges + numEdgesInSubtree
				}
			
				edges[nonEmptyAttri+1,3] = numNodes + 1
				nonEmptyAttri = nonEmptyAttri + 1
				
				numNodes = numNodes + numNodesInSubtree
			}
		}
	}
  
  return ( list(a=nodes, b=edges) );
}

X = readMM(paste(args[1], "X.mtx", sep=""));
y = readMM(paste(args[1], "y.mtx", sep=""));

n = nrow(X)
m = ncol(X)

minsplit = 2


X_subset = matrix(1, n, 1)
attributes = matrix(1, m, 1)
# recoding inputs
featureCorrections = as.vector(1 - colMins(as.matrix(X)))
onesMat = matrix(1, n, m)

X = onesMat %*% diag(featureCorrections) + X
labelCorrection = 1 - min(y)
y = y + labelCorrection + 0

tmpRet <- id3_learn(X, y, X_subset, attributes, minsplit)
nodes = as.matrix(tmpRet$a)
edges = as.matrix(tmpRet$b)

# decoding outputs
nodes[,2] = nodes[,2] - labelCorrection * (nodes[,1] == -1)
for(i3 in 1:nrow(edges)){
	e_parent = as.numeric(edges[i3,1])
	parent_feature = as.numeric(nodes[e_parent,1])
	correction = as.numeric(featureCorrections[parent_feature])
	edges[i3,2] = edges[i3,2] - correction
}

writeMM(as(nodes,"CsparseMatrix"), paste(args[2],"nodes", sep=""), format = "text")
writeMM(as(edges,"CsparseMatrix"), paste(args[2],"edges", sep=""), format = "text")

