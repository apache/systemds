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

package org.apache.sysds.runtime.iogen;

import java.util.HashSet;

public class DelimiterTrie {
	private DelimiterTrieNode root;

	public DelimiterTrie() {
		root = new DelimiterTrieNode();
	}

	private String intersect(String str1, String str2) {
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < Math.min(str1.length(), str2.length()); i++) {
			if(str1.charAt(i) == str2.charAt(i))
				sb.append(str1.charAt(i));
			else
				break;
		}
		if(sb.length() == 0)
			return null;
		else
			return sb.toString();
	}

	private TrieNodeResult getSubNode(DelimiterTrieNode current, String delim) {
		for(String key : current.getChildren().keySet()) {
			String insec = intersect(key, delim);
			if(insec != null)
				return new TrieNodeResult(current.getChildren().get(key), insec, key);
		}
		return null;
	}

	public void insert(String delim) {
		DelimiterTrieNode current = root;
		String remaindKeyDelim;
		String currentDelim = delim;
		TrieNodeResult trieNodeResult;
		do {
			trieNodeResult = getSubNode(current, currentDelim);
			if(trieNodeResult == null) {
				DelimiterTrieNode newNode = new DelimiterTrieNode();
				current.getChildren().put(currentDelim, newNode);
			}
			else {
				currentDelim = currentDelim.substring(trieNodeResult.intersect.length());
				remaindKeyDelim = trieNodeResult.nodeKey.substring(trieNodeResult.intersect.length());
				int cwl = currentDelim.length();
				int rkwl = remaindKeyDelim.length();

				if(cwl == 0 && rkwl > 0) {
					DelimiterTrieNode newNode = new DelimiterTrieNode();

					DelimiterTrieNode updateNode = new DelimiterTrieNode();
					updateNode.setChildren(trieNodeResult.trieNode.getChildren());

					// Add Update Node
					newNode.getChildren().put(remaindKeyDelim, updateNode);

					// Add New Node
					current.getChildren().put(trieNodeResult.intersect, newNode);

					// Remove old node
					current.getChildren().remove(trieNodeResult.nodeKey);

				}
				else if(rkwl == 0) {
					current = trieNodeResult.trieNode;
				}
				else {
					DelimiterTrieNode newNode = new DelimiterTrieNode();

					DelimiterTrieNode updateNode = new DelimiterTrieNode();
					updateNode.setChildren(trieNodeResult.trieNode.getChildren());

					// Add Update Node
					newNode.getChildren().put(remaindKeyDelim, updateNode);

					// Add New Node
					current.getChildren().put(trieNodeResult.intersect, newNode);

					// Remove old node
					current.getChildren().remove(trieNodeResult.nodeKey);

					// Add New Delim remaind
					DelimiterTrieNode newDelimNode = new DelimiterTrieNode();
					newNode.getChildren().put(currentDelim, newDelimNode);
					break;
				}
			}

		}
		while(trieNodeResult != null && currentDelim.length() > 0);
	}

	public String getShortestDelim(int minsize) {
		// Check the possibility of the shortest delim
		boolean flag = true;
		DelimiterTrieNode current = root;
		StringBuilder sb = new StringBuilder();
		do {
			int currentChildCount = current.getChildren().size();
			if(currentChildCount == 0)
				break;
			else if(currentChildCount != 1)
				flag = false;
			else {
				String key = current.getChildren().keySet().iterator().next();
				sb.append(key);
				current = current.getChildren().get(key);
			}
		}
		while(flag);
		if(flag) {
			String allDelim = sb.toString();
			int allDelimLength = allDelim.length();
			HashSet<String> delimSet = new HashSet<>();
			for(int i=1; i<=minsize; i++){
				delimSet.clear();
				for(int j=0; j<allDelim.length();j+=i){
					delimSet.add(allDelim.substring(j,Math.min(j+i, allDelimLength)));
				}
				if(delimSet.size() == 1)
					break;
			}
		}
		return null;
	}

	private class TrieNodeResult {
		DelimiterTrieNode trieNode;
		String intersect;
		String nodeKey;

		public TrieNodeResult(DelimiterTrieNode trieNode, String intersect, String nodeKey) {
			this.trieNode = trieNode;
			this.intersect = intersect;
			this.nodeKey = nodeKey;
		}

		public DelimiterTrieNode getTrieNode() {
			return trieNode;
		}

		public void setTrieNode(DelimiterTrieNode trieNode) {
			this.trieNode = trieNode;
		}

		public String getIntersect() {
			return intersect;
		}

		public void setIntersect(String intersect) {
			this.intersect = intersect;
		}
	}
}
