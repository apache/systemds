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

import java.util.ArrayList;
import java.util.HashSet;

public class ColKeyTrie {

	private ColKeyTrieNode rootPrefixKeys;
	private ColKeyTrieNode rootSuffixKeys;
	private ArrayList<ArrayList<String>> prefixKeyPattern;

	public ColKeyTrie(ArrayList<ArrayList<String>> prefixKeyPattern) {
		this.rootPrefixKeys = new ColKeyTrieNode("RootPrefixKeys");
		this.rootSuffixKeys = new ColKeyTrieNode("RootSuffixKeys");
		this.prefixKeyPattern = prefixKeyPattern;
	}

	public ColKeyTrie(String colDelim) {
		this.rootPrefixKeys = new ColKeyTrieNode("RootPrefixKeys");
		this.rootSuffixKeys = new ColKeyTrieNode("RootSuffixKeys");
		this.prefixKeyPattern = null;

		ColKeyTrieNode newNode;
		newNode = new ColKeyTrieNode(colDelim);
		this.rootPrefixKeys.getChildren().put(colDelim, newNode);
	}

	public void insertPrefixKeys(ArrayList<String> keys) {
		this.insertKeys(keys, rootPrefixKeys);
	}

	public void insertSuffixKeys(ArrayList<String> keys) {
		this.insertKeys(keys, rootSuffixKeys);
	}

	public void insertSuffixKeys(char[] keys) {
		ArrayList<String> charList = new ArrayList<>();
		for(Character ch : keys)
			charList.add(ch.toString());
		this.insertKeys(charList, rootSuffixKeys);
	}

	public void setAPrefixPath(ArrayList<String> keys) {
		ColKeyTrieNode currentNode = rootPrefixKeys;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode = currentNode.getChildren().get(key);
				currentNode.setCheck(true);
			}
		}
	}

	private void insertKeys(ArrayList<String> keys, ColKeyTrieNode root) {
		ColKeyTrieNode currentNode = root;
		int index = 0;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode.countPP();
				currentNode = currentNode.getChildren().get(key);
				currentNode.countPP();
				index++;
			}
			else
				break;
		}

		ColKeyTrieNode newNode;
		for(int i = index; i < keys.size(); i++) {
			newNode = new ColKeyTrieNode(keys.get(i));
			currentNode.getChildren().put(keys.get(i), newNode);
			currentNode = newNode;
		}
	}

	public ArrayList<ArrayList<String>> getPrefixKeyPatterns() {
		if(this.prefixKeyPattern!=null)
			return prefixKeyPattern;
		else
			return getKeyPatterns(rootPrefixKeys);
	}

	public ArrayList<ArrayList<String>> getSuffixKeyPatterns() {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		for(String k : rootSuffixKeys.getChildren().keySet()) {
			ColKeyTrieNode node = rootSuffixKeys.getChildren().get(k);
			ArrayList<String> nk = new ArrayList<>();
			nk.add(k);
			int maxCount = node.getCount();
			getKeyPatterns2(node, result, nk, maxCount);
		}
		return result;
	}

	private ArrayList<ArrayList<String>> getKeyPatterns(ColKeyTrieNode root) {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		getKeyPatterns(root, result, new ArrayList<>());
		return result;
	}

	private void getKeyPatterns(ColKeyTrieNode node, ArrayList<ArrayList<String>> result, ArrayList<String> nodeKeys) {

		if(node.getChildren().size() == 0) {
			result.add(nodeKeys);
			nodeKeys = new ArrayList<>();
		}
		else {
			for(String k : node.getChildren().keySet()) {
				ColKeyTrieNode child = node.getChildren().get(k);
				ArrayList<String> tmpKeys = new ArrayList<>();
				tmpKeys.addAll(nodeKeys);
				tmpKeys.add(k);
				getKeyPatterns(child, result, tmpKeys);
			}
		}
	}

	private void getKeyPatterns2(ColKeyTrieNode node, ArrayList<ArrayList<String>> result, ArrayList<String> nodeKeys,
		int maxCount) {

		if(node.getChildren().size() == 1 && node.getCount() == maxCount) {
			String k = node.getChildren().keySet().iterator().next();
			ColKeyTrieNode child = node.getChildren().get(k);
			ArrayList<String> tmpKeys = new ArrayList<>();
			tmpKeys.addAll(nodeKeys);
			tmpKeys.add(k);
			getKeyPatterns2(child, result, tmpKeys, maxCount);
		}
		else
			result.add(nodeKeys);

	}

	public void insertPrefixKeysConcurrent(HashSet<String> keys) {
		insertPrefixKeysConcurrent(rootPrefixKeys, keys);
	}

	private void insertPrefixKeysConcurrent(ColKeyTrieNode node, HashSet<String> keys) {
		if(node.getChildren().size() == 0) {
			for(String k : keys) {
				ColKeyTrieNode newNode = new ColKeyTrieNode(k);
				node.getChildren().put(k, newNode);
			}
		}
		else {
			for(String childKey : node.getChildren().keySet()) {
				ColKeyTrieNode child = node.getChildren().get(childKey);
				insertPrefixKeysConcurrent(child, keys);
			}
		}
	}
}
