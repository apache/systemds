/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.runtime.privacy.FineGrained;

import java.util.Map;

import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;

/**
 * Basic R Tree implementation of fine-grained privacy. 
 * THE IMPLEMENTATION IS NOT DONE!
 */
public class FineGrainedPrivacyRTree implements FineGrainedPrivacy {
	private InternalNode rootNode;
	private int maxChildren; //TODO: Activate this to determine the maximum number of children per node. 

	public abstract class Node {

	}

	public class RootNode extends Node {
		public Node[] childNodes;
	}

	public class InternalNode extends Node {
		public DataRange boundingBox;
		public Node[] childNodes;

		/**
		 * Use this constructor for root node.
		 */
		public InternalNode(){}

		public InternalNode(DataRange boundingBox, Node[] childNodes){
			this.boundingBox = boundingBox;
			this.childNodes = childNodes;
		}
	}

	public class LeafNode extends Node {
		public DataRange constraintRange;
		public PrivacyLevel constraintPrivacyLevel;

		public LeafNode(DataRange constraintRange, PrivacyLevel constraintPrivacyLevel){
			this.constraintRange = constraintRange;
			this.constraintPrivacyLevel = constraintPrivacyLevel;
		}
	}

	public FineGrainedPrivacyRTree(){
		rootNode = new InternalNode();
	}

	@Override
	public void put(DataRange dataRange, PrivacyLevel privacyLevel) {
		LeafNode newNode = new LeafNode(dataRange, privacyLevel);
		if (rootNode.childNodes == null || rootNode.childNodes.length == 0)
			rootNode.childNodes = new Node[]{newNode};
		else
		{
			put(rootNode, newNode);
		}
	}

	private void put(Node currentNode, LeafNode newNode){
		if ( currentNode instanceof InternalNode ){
			InternalNode currentInternalNode = ((InternalNode) currentNode);
			if ( currentInternalNode.childNodes != null && currentInternalNode.childNodes.length > 0 && currentInternalNode.childNodes[0] instanceof InternalNode ){
				Node closestChild = findClosestChild((InternalNode[])currentInternalNode.childNodes, newNode);
				put(closestChild, newNode);
			}
			
		}
		
	}

	private Node findClosestChild(InternalNode[] nodes, LeafNode newNode){
		long[] scores = getMinimumAreaEnlargmentScores(nodes, newNode);
		
		// Compare the scores for proximity to newNode
		int minIndex = -1;
		long minValue = Long.MAX_VALUE;
		for (int i = 0; i < scores.length; i++){
			if ( scores[i] > minValue){
				minIndex = i;
				minValue = scores[i];
			}
		}

		return nodes[minIndex];
	}
	
	private long[] getMinimumAreaEnlargmentScores(InternalNode[] nodes, LeafNode newNode){
		long[] scores = new long[nodes.length];
		for ( int i = 0; i < nodes.length; i++ ){
			// Retrieve a score for proximity to newNode
			scores[i] = getMAEScore(nodes[i], newNode);
		}
		return scores;
	}

	private long getMAEScore(InternalNode node, LeafNode newNode){

		long[] beginDims = node.boundingBox.getBeginDims();
		long[] endDIms = node.boundingBox.getEndDims();
		long[] newNodeBeginDims = newNode.constraintRange.getBeginDims();
		long[] newNodeEndDims = newNode.constraintRange.getEndDims();
		long score = Long.MAX_VALUE;
		for(int i = 0; i < beginDims.length; i++){
			//TODO: Calculate minimum area enlargement score
		}
		return score;
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevel(DataRange searchRange) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Map<DataRange, PrivacyLevel> getPrivacyLevelOfElement(long[] searchIndex) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public DataRange[] getDataRangesOfPrivacyLevel(PrivacyLevel privacyLevel) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void removeAllConstraints() {
		// TODO Auto-generated method stub

	}
	
}