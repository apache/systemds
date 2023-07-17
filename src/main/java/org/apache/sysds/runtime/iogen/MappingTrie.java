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

import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.matrix.data.Pair;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class MappingTrie {

	private MappingTrieNode root;
	private int keyLevel;
	private boolean inALine;
	private int windowSize = 100;

	public MappingTrie() {
		this.root = new MappingTrieNode(MappingTrieNode.Type.INNER);
		this.keyLevel = 0;
		this.inALine = true;
	}

	public void setInALine(boolean inALine) {
		this.inALine = inALine;
	}

	public void insert(String word, int rowIndex) {
		ArrayList<Integer> tmpList = new ArrayList<>();
		tmpList.add(rowIndex);
		this.insert(word, tmpList);
	}

	public void reverseInsert(String word, int rowIndex) {
		ArrayList<Integer> tmpList = new ArrayList<>();
		tmpList.add(rowIndex);
		this.insert(new StringBuilder(word).reverse().toString(), tmpList);
	}

	public void insert(String word, ArrayList<Integer> rowIndexes) {
		MappingTrieNode newNode;
		if(root.getChildren().containsKey(word))
			newNode = root.getChildren().get(word);
		else
			newNode = new MappingTrieNode();
		newNode.addRowIndex(rowIndexes);
		root.getChildren().put(word, newNode);
	}

	public MappingTrieNode getFistMultiChildNode(MappingTrieNode node) {

		if(node.getNodeType() == MappingTrieNode.Type.INNER && node.getChildren().size() == 1) {
			String nkey = node.getChildren().keySet().iterator().next();
			if(node.getChildren().get(nkey).getRowIndexes().size() > 1)
				return node;
		}
		if(node.getChildren().size() == 1 && node.getNodeType() != MappingTrieNode.Type.END)
			return getFistMultiChildNode(node.getChildren().get(node.getChildren().keySet().iterator().next()));
		else
			return node;
	}

	public void insertKeys(ArrayList<String> keys) {
		MappingTrieNode currentNode = root;
		int index = 0;
		for(String key : keys) {
			if(currentNode.getChildren().containsKey(key)) {
				currentNode = currentNode.getChildren().get(key);
				index++;
			}
			else
				break;
		}

		MappingTrieNode newNode;
		for(int i = index; i < keys.size(); i++) {
			newNode = new MappingTrieNode();
			currentNode.getChildren().put(keys.get(i), newNode);
			currentNode = newNode;
		}
	}

	public Set<String> getAllSubStringsOfStringContainIntersect(String str, BitSet bitSet) {
		HashSet<String> result = new HashSet<>();
		StringBuilder sb = new StringBuilder();
		for(int i = 0; i < bitSet.size(); i++) {
			if(bitSet.get(i))
				sb.append(str.charAt(i));
			else if(sb.length() > 0) {
				getAllSubStrings(result, sb);
				sb = new StringBuilder();
			}
		}
		if(sb.length() > 0) {
			getAllSubStrings(result, sb);
		}

		return result;
	}

	private void getAllSubStrings(HashSet<String> result, StringBuilder sb) {
		if(sb.length() == 1)
			result.add(sb.toString());
		else {
			for(int j = 1; j <= Math.min(sb.length(), windowSize); j++) {
				for(int k = 0; k <= sb.length() - j; k++) {
					result.add(sb.substring(k, k + j));
				}
			}
		}
	}

	public String getIntersectOfChildren(MappingTrieNode node) {
		if(node.getNodeType() == MappingTrieNode.Type.END || node.getChildren().size() == 0)
			return null;
		else {
			Set<String> keys = node.getChildren().keySet();
			if(keys.size() == 1) {
				String[] splitText = keys.iterator().next().split(Lop.OPERAND_DELIMITOR, -1);
				String str = splitText[0];
				if(str.length() == 0 && splitText.length > 1)
					str = splitText[1];
				return String.valueOf(str.charAt(0));
			}

			Set<String> newKeys = new HashSet<>();
			for(String k : keys) {
				String[] splitText = k.split(Lop.OPERAND_DELIMITOR, -1);
				String str = splitText[0];
				if(str.length() == 0 && splitText.length > 1)
					str = splitText[1];
				newKeys.add(str);
			}
			keys = newKeys;

			boolean flag = false;
			int maxKeyLength = 0;
			Set<Character> intersections = null;
			for(String k : keys) {
				if(flag) {
					intersections.retainAll(k.chars().mapToObj(c -> (char) c).collect(Collectors.toSet()));
				}
				else {
					intersections = k.chars().mapToObj(c -> (char) c).collect(Collectors.toSet());
					flag = true;
				}

				// set max length of key
				maxKeyLength = Math.max(maxKeyLength, k.length());
			}
			if(intersections == null || intersections.size() == 0)
				return null;
			else {
				Set<String> subStringIntersection = new HashSet<>();
				boolean subStringIntersectionFlag = false;
				for(String k : keys) {
					BitSet bitSets = new BitSet(maxKeyLength);
					int i = 0;
					for(Character character : k.toCharArray()) {
						if(intersections.contains(character))
							bitSets.set(i);
						i++;
					}
					if(subStringIntersectionFlag) {
						subStringIntersection.retainAll(getAllSubStringsOfStringContainIntersect(k, bitSets));
					}
					else {
						subStringIntersection = getAllSubStringsOfStringContainIntersect(k, bitSets);
						subStringIntersectionFlag = true;
					}
				}
				if(subStringIntersection.size() == 1) {
					return subStringIntersection.iterator().next();
				}
				else {
					ArrayList<String> sortedList = (ArrayList<String>) subStringIntersection.stream()
						.sorted((o1, o2) -> o2.length() - o1.length()).collect(Collectors.toList());

					for(String ssi : sortedList) {
						if(keyLevel == 0 && inALine) {
							boolean flagBest = true;
							for(String k : keys) {
								if(!k.startsWith(ssi)) {
									flagBest = false;
									break;
								}
							}
							if(flagBest)
								return ssi;
						}
						else {
							int lastCount = 0;
							for(String k : keys) {
								int beginPos = 0;
								int count = 0;
								do {
									int index = k.indexOf(ssi, beginPos);
									if(index != -1) {
										count++;
										beginPos = index + ssi.length();
									}
									else
										break;
								}
								while(true);
//								if(count==1)
//									lastCount = 1;
//								else {
//									lastCount = 0;
//									break;
//								}
								if(lastCount != 0 && lastCount != count) {
									lastCount = 0;
									break;
								}
								else if(lastCount == 0)
									lastCount = count;

							}
							if(lastCount != 0)
								return ssi;
						}
					}
					return null;
				}
			}
		}
	}

	public MappingTrieNode getRoot() {
		return root;
	}

	public boolean reConstruct() {
		MappingTrieNode node = getFistMultiChildNode(root);
		String intersect = getIntersectOfChildren(node);

		// prune nodes if they don't have any intersect char
		if(intersect == null) {
			node.getChildren().clear();
			node.setNodeType(MappingTrieNode.Type.END);
			return false;
		}
		else {
			MappingTrieNode.Type intersectNodeType = MappingTrieNode.Type.INNER;
			MappingTrie intersectTrie = new MappingTrie();
			ArrayList<Integer> intersectRowIndexes = new ArrayList<>();

			for(String k : node.getChildren().keySet()) {
				String key = k.substring(k.indexOf(intersect) + intersect.length());
				if(key.length() > 0 && !key.equals(Lop.OPERAND_DELIMITOR)) {
					intersectTrie.insert(key, node.getChildren().get(k).getRowIndexes());
					intersectRowIndexes.addAll(node.getChildren().get(k).getRowIndexes());
				}
				else
					intersectNodeType = MappingTrieNode.Type.END;
			}

			// clear the node children
			node.getChildren().clear();

			// create an IGNORE node type and add it to the tree
			MappingTrieNode ignoreNode = new MappingTrieNode(MappingTrieNode.Type.IGNORE);

			node.getChildren().put(null, ignoreNode);

			// create and add intersection node
			MappingTrieNode intersectionNode = new MappingTrieNode(intersectNodeType);
			intersectionNode.setChildren(intersectTrie.root.getChildren());
			intersectionNode.setRowIndexes(intersectRowIndexes);
			ignoreNode.getChildren().put(intersect, intersectionNode);

			keyLevel++;

			return true;
		}
	}

	public ArrayList<ArrayList<String>> getAllSequentialKeys() {
		ArrayList<ArrayList<Pair<String, ArrayList<Integer>>>> result = new ArrayList<>();
		getAllSequentialKeys(root, result, new ArrayList<>());

		// orders
		ArrayList<Pair<Integer, Integer>> indexOrder = new ArrayList<>();
		int index = 0;
		for(ArrayList<Pair<String, ArrayList<Integer>>> k : result) {
			int level = 0;
			for(Pair<String, ArrayList<Integer>> n : k) {
				if(n.getKey() != null) {
					if(level == keyLevel - 1 || keyLevel == 0) {
						indexOrder.add(new Pair<>(index, n.getValue().size()));
						break;
					}
					level++;
				}
			}
			index++;
		}

		List<Pair<Integer, Integer>> sortedList = indexOrder.stream()
			.sorted((o1, o2) -> o2.getValue().compareTo(o1.getValue())).collect(Collectors.toList());

		ArrayList<ArrayList<String>> keys = new ArrayList<>();
		for(Pair<Integer, Integer> p : sortedList) {
			ArrayList<Pair<String, ArrayList<Integer>>> k = result.get(p.getKey());
			ArrayList<String> kl = new ArrayList<>();
			int level = 0;
			for(Pair<String, ArrayList<Integer>> n : k)
				if(n.getKey() != null) {
					if(level < keyLevel || keyLevel == 0) {
						String[] splitText = n.getKey().split(Lop.OPERAND_DELIMITOR, -1);
						String str = splitText[0];
						if(str.length() == 0 && splitText.length > 1)
							str = splitText[1];

						kl.add(str);
						level++;
					}
					else
						break;
				}

			keys.add(kl);
		}
		ArrayList<ArrayList<String>> distinctKeys = new ArrayList<>();
		HashSet<Integer> markedIndexes = new HashSet<>();
		ArrayList<String> selected;
		for(int i = 0; i < keys.size(); i++) {
			if(markedIndexes.contains(i))
				continue;
			else {
				selected = keys.get(i);
				markedIndexes.add(i);
				distinctKeys.add(selected);
			}
			for(int j = i + 1; j < keys.size(); j++) {
				if(!markedIndexes.contains(j)) {
					boolean flag = true;
					for(int k = 0; k < selected.size(); k++) {
						if(!selected.get(k).equals(keys.get(j).get(k))) {
							flag = false;
							break;
						}
					}
					if(flag) {
						markedIndexes.add(j);
					}
				}
			}
		}

		// revert list and values of list
		for(ArrayList<String> l : distinctKeys) {
			Collections.reverse(l);
			for(int i = 0; i < l.size(); i++) {
				l.set(i, new StringBuilder(l.get(i)).reverse().toString());
			}
		}
		return distinctKeys;
	}

	private void getAllSequentialKeys(MappingTrieNode node,
		ArrayList<ArrayList<Pair<String, ArrayList<Integer>>>> result,
		ArrayList<Pair<String, ArrayList<Integer>>> nodeKeys) {

		if(node.getNodeType() == MappingTrieNode.Type.END) {
			result.add(nodeKeys);
			nodeKeys = new ArrayList<>();
		}
		else {
			for(String k : node.getChildren().keySet()) {
				MappingTrieNode child = node.getChildren().get(k);
				ArrayList<Pair<String, ArrayList<Integer>>> tmpKeys = new ArrayList<>();
				tmpKeys.addAll(nodeKeys);
				tmpKeys.add(new Pair<>(k, child.getRowIndexes()));
				getAllSequentialKeys(child, result, tmpKeys);
			}
		}
	}

	public void setWindowSize(int windowSize) {
		this.windowSize = windowSize;
	}
}
