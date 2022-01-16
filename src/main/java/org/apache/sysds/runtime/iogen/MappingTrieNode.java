package org.apache.sysds.runtime.iogen;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class MappingTrieNode {

	public enum Type {
		INNER, END, IGNORE;
		@Override public String toString() {
			return this.name().toUpperCase();
		}
	}

	private Map<String, MappingTrieNode> children;
	private Type nodeType;
	private ArrayList<Integer> rowIndexes;

	public MappingTrieNode(Type nodeType) {
		this.nodeType = nodeType;
		children = new HashMap<>();
		rowIndexes = new ArrayList<>();
	}

	public MappingTrieNode() {
		this.nodeType = Type.END;
		children = new HashMap<>();
		rowIndexes = new ArrayList<>();
	}

	public Map<String, MappingTrieNode> getChildren() {
		return children;
	}

	public void setChildren(Map<String, MappingTrieNode> children) {
		this.children = children;
	}

	public Type getNodeType() {
		return nodeType;
	}

	public void setNodeType(Type nodeType) {
		this.nodeType = nodeType;
	}

	public void addRowIndex(int rowIndex) {
		rowIndexes.add(rowIndex);
	}

	public void addRowIndex(ArrayList<Integer> rowIndexes) {
		this.rowIndexes.addAll(rowIndexes);
	}

	public void setRowIndexes(ArrayList<Integer> rowIndexes) {
		this.rowIndexes = rowIndexes;
	}

	public ArrayList<Integer> getRowIndexes() {
		return rowIndexes;
	}
}
