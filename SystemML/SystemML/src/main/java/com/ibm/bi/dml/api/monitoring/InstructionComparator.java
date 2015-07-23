package com.ibm.bi.dml.api.monitoring;

import java.util.Comparator;
import java.util.HashMap;

public class InstructionComparator implements Comparator<String>{

	HashMap<String, Long> instructionCreationTime;
	public InstructionComparator(HashMap<String, Long> instructionCreationTime) {
		this.instructionCreationTime = instructionCreationTime;
	}
	@Override
	public int compare(String o1, String o2) {
		try {
			return instructionCreationTime.get(o1).compareTo(instructionCreationTime.get(o2));
		}
		catch(Exception e) {
			return -1;
		}
	}

}
