package org.apache.sysds.runtime.iogen;

import java.util.ArrayList;
import java.util.HashSet;

public class ColumnProperties {

	private ArrayList<ArrayList<String>> keyPatterns;
	private HashSet<String> endWithValueString;
	private ArrayList<ArrayList<String>> nextToPatterns;

	public ArrayList<ArrayList<String>> getKeyPatterns() {
		return keyPatterns;
	}

	public void setKeyPatterns(ArrayList<ArrayList<String>> keyPatterns) {
		this.keyPatterns = keyPatterns;
	}

	public HashSet<String> getEndWithValueString() {
		return endWithValueString;
	}

	public void setEndWithValueString(HashSet<String> endWithValueString) {
		this.endWithValueString = endWithValueString;
	}

	public ArrayList<ArrayList<String>> getNextToPatterns() {
		return nextToPatterns;
	}

	public void setNextToPatterns(ArrayList<ArrayList<String>> nextToPatterns) {
		this.nextToPatterns = nextToPatterns;
	}
}
