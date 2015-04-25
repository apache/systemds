package com.ibm.bi.dml.runtime.instructions.spark.data;

import java.io.Serializable;

public class CountLinesInfo implements Serializable {
	private static final long serialVersionUID = 4178309746487858987L;
	private long numLines;
	private long expectedNumColumns;
	public long getNumLines() {
		return numLines;
	}
	public void setNumLines(long numLines) {
		this.numLines = numLines;
	}
	public long getExpectedNumColumns() {
		return expectedNumColumns;
	}
	public void setExpectedNumColumns(long expectedNumColumns) {
		this.expectedNumColumns = expectedNumColumns;
	}
	
}
