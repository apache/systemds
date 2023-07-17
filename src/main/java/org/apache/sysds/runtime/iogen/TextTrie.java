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

import com.google.gson.Gson;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class TextTrie {
	private TextTrieNode root;

	TextTrie() {
		root = new TextTrieNode();
	}

	public void reverseInsert(String word, int rowIndex){
		TextTrieNode current = root;
		for(int i = word.length() -1; i>=0; i-- ) {
			current = current.getChildren().computeIfAbsent(word.charAt(i), c -> new TextTrieNode());
			current.addRowIndex(rowIndex);
		}
		current.setEndOfWord(true);
	}

	public void insert(String word, int rowIndex) {
		TextTrieNode current = root;
		for(char l : word.toCharArray()) {
			current = current.getChildren().computeIfAbsent(l, c -> new TextTrieNode());
			current.addRowIndex(rowIndex);
		}
		current.setEndOfWord(true);
	}

	public TextTrieNode containsString(String word) {
		TextTrieNode current = root;
		for(int i = 0; i < word.length(); i++) {
			char ch = word.charAt(i);
			TextTrieNode node = current.getChildren().get(ch);
			if(node == null) {
				return null;
			}
			current = node;
		}
		return current;
	}

	public int containsStringAndSet(String word) {
		TextTrieNode result = containsString(word);
		int rowIndex = -1;
		if(result != null) {
			rowIndex = result.getRowIndex();
			if(rowIndex != -1)
				result.setRowIndexUsed(rowIndex);
		}
		return rowIndex;
	}

	public ArrayList<Pair<String, Set<Integer>>> getAllKeys(){
		ArrayList<Pair<String, Set<Integer>>> result = new ArrayList<>();
		ArrayList<Key> allKeys = new ArrayList<>();
		getAllKeys(root, allKeys, new Key(new StringBuilder(), new ArrayList<>()));

		Comparator<Key> compare = Comparator.comparing(Key::getIndexSetSize).thenComparing(Key::getKeyLength).reversed();
		List<Key> sortedKeys = allKeys.stream().sorted(compare).collect(Collectors.toList());

		for(Key k: sortedKeys){
			result.add(new Pair<>(k.getKey().toString(), k.getIndexSet()));
		}
		return result;
	}

	private void getAllKeys(TextTrieNode node, ArrayList<Key> result, Key curKey){
		if(node.getChildren().size() == 0)
			return;
		else {
			for(Character k: node.getChildren().keySet()){
				TextTrieNode child = node.getChildren().get(k);
				ArrayList<Integer> tList = new ArrayList<>();
				tList.addAll(child.getRowIndexes());
				Key key = new Key( new StringBuilder(curKey.getKey()).append(k), tList);
				result.add(key);
				getAllKeys(child, result, key);
			}
		}
	}

	private class Key{
		private StringBuilder key;
		private ArrayList<Integer> rowIndexes;
		private int keyLength;
		private Set<Integer> indexSet;
		private int indexSetSize;

		public Key(StringBuilder key, ArrayList<Integer> rowIndexes) {
			this.key = key;
			this.rowIndexes = rowIndexes;
			this.keyLength = key.length();
			this.indexSet = new HashSet<>();
			this.indexSet.addAll(rowIndexes);
			this.indexSetSize = this.indexSet.size();
		}

		public StringBuilder getKey() {
			return key;
		}


		public void setKey(StringBuilder key) {
			this.key = key;
		}

		public ArrayList<Integer> getRowIndexes() {
			return rowIndexes;
		}

		public void setRowIndexes(ArrayList<Integer> rowIndexes) {
			this.rowIndexes = rowIndexes;
		}

		public int getKeyLength() {
			return keyLength;
		}

		public Set<Integer> getIndexSet() {
			return indexSet;
		}

		public int getIndexSetSize() {
			return indexSetSize;
		}

		public void print(){
			Gson gson = new Gson();
			System.out.println(key.toString()+" "+gson.toJson(this.indexSet));
		}
	}
}

