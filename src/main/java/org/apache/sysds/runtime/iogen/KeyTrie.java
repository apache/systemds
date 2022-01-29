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
import java.util.Collections;
import java.util.HashSet;

public class KeyTrie {

	private KeyTrieNode rootPrefixKeys;
	private KeyTrieNode rootSuffixKeys;
	private ArrayList<ArrayList<String>> prefixKeyPattern;

	public KeyTrie() {
		this.rootPrefixKeys = new KeyTrieNode("RootPrefixKeys");
		this.rootSuffixKeys = new KeyTrieNode("RootSuffixKeys");
		this.prefixKeyPattern = null;
	}

	public KeyTrie(ArrayList<ArrayList<String>> prefixKeyPattern) {
		this.rootPrefixKeys = new KeyTrieNode("RootPrefixKeys");
		this.rootSuffixKeys = new KeyTrieNode("RootSuffixKeys");
		this.prefixKeyPattern = prefixKeyPattern;
	}

	public KeyTrie(String colDelim) {
		this.rootPrefixKeys = new KeyTrieNode("RootPrefixKeys");
		this.rootSuffixKeys = new KeyTrieNode("RootSuffixKeys");
		this.prefixKeyPattern = null;

		KeyTrieNode newNode;
		newNode = new KeyTrieNode(colDelim);
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
		KeyTrieNode currentNode = rootPrefixKeys;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode = currentNode.getChildren().get(key);
				currentNode.setCheck(true);
			}
		}
	}

	private void insertKeys(ArrayList<String> keys, KeyTrieNode root) {
		KeyTrieNode currentNode = root;
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

		KeyTrieNode newNode;
		for(int i = index; i < keys.size(); i++) {
			newNode = new KeyTrieNode(keys.get(i));
			currentNode.getChildren().put(keys.get(i), newNode);
			currentNode = newNode;
		}
	}

	public ArrayList<ArrayList<String>> getPrefixKeyPatterns() {
		if(this.prefixKeyPattern != null)
			return prefixKeyPattern;
		else
			return getKeyPatterns(rootPrefixKeys);
	}

	public ArrayList<ArrayList<String>> getReversePrefixKeyPatterns() {
		if(this.prefixKeyPattern != null)
			return prefixKeyPattern;
		else {
			ArrayList<ArrayList<String>> kps = getKeyPatterns(rootPrefixKeys);
			for(ArrayList<String> l : kps) {
				Collections.reverse(l);
				for(int i = 0; i < l.size(); i++) {
					l.set(i, new StringBuilder(l.get(i)).reverse().toString());
				}
			}
			return kps;
		}
	}

	public ArrayList<ArrayList<String>> getSuffixKeyPatterns() {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		for(String k : rootSuffixKeys.getChildren().keySet()) {
			KeyTrieNode node = rootSuffixKeys.getChildren().get(k);
			ArrayList<String> nk = new ArrayList<>();
			nk.add(k);
			int maxCount = node.getCount();
			getSuffixKeyPatterns(node, result, nk, maxCount);
		}
		return result;
	}

	public HashSet<String> getFirstSuffixKeyPatterns() {
		ArrayList<ArrayList<String>> suffixKeyPattern = getSuffixKeyPatterns();
		HashSet<String> suffixString = new HashSet<>();
		for(ArrayList<String> kp : suffixKeyPattern) {
			suffixString.add(kp.get(0));
		}
		return suffixString;
	}

	private ArrayList<ArrayList<String>> getKeyPatterns(KeyTrieNode root) {
		ArrayList<ArrayList<String>> result = new ArrayList<>();
		getKeyPatterns(root, result, new ArrayList<>());
		return result;
	}

	private void getKeyPatterns(KeyTrieNode node, ArrayList<ArrayList<String>> result, ArrayList<String> nodeKeys) {

		if(node.getChildren().size() == 0) {
			result.add(nodeKeys);
			nodeKeys = new ArrayList<>();
		}
		else {
			for(String k : node.getChildren().keySet()) {
				KeyTrieNode child = node.getChildren().get(k);
				ArrayList<String> tmpKeys = new ArrayList<>();
				tmpKeys.addAll(nodeKeys);
				tmpKeys.add(k);
				getKeyPatterns(child, result, tmpKeys);
			}
		}
	}

	private void getSuffixKeyPatterns(KeyTrieNode node, ArrayList<ArrayList<String>> result, ArrayList<String> nodeKeys,
		int maxCount) {

		if(node.getChildren().size() == 1 && node.getCount() == maxCount) {
			String k = node.getChildren().keySet().iterator().next();
			KeyTrieNode child = node.getChildren().get(k);
			ArrayList<String> tmpKeys = new ArrayList<>();
			tmpKeys.addAll(nodeKeys);
			tmpKeys.add(k);
			getSuffixKeyPatterns(child, result, tmpKeys, maxCount);
		}
		else
			result.add(nodeKeys);

	}

	public void insertPrefixKeysConcurrent(HashSet<String> keys) {
		insertPrefixKeysConcurrent(rootPrefixKeys, keys);
		ArrayList<ArrayList<String>> ss = getPrefixKeyPatterns();
	}

	private void insertPrefixKeysConcurrent(KeyTrieNode node, HashSet<String> keys) {
		if(node.getChildren().size() == 0) {
			for(String k : keys) {
				KeyTrieNode newNode = new KeyTrieNode(k);
				node.getChildren().put(k, newNode);
			}
		}
		else {
			for(String childKey : node.getChildren().keySet()) {
				KeyTrieNode child = node.getChildren().get(childKey);
				insertPrefixKeysConcurrent(child, keys);
			}
		}
	}

	public void setPrefixKeyPattern(ArrayList<ArrayList<String>> prefixKeyPattern) {
		this.prefixKeyPattern = prefixKeyPattern;
	}
}
