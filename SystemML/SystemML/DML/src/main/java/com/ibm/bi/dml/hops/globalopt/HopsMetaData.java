/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.Hop;

/**
 * Encapsulates all the meta data for one particular {@link Hops} instance. 
 * (1) block dimensions
 * (2) data dimensions
 */
public class HopsMetaData 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Hop operator;
	private boolean reblockRequired;
	public static final long UNDEFINED = -1L;
	//TODO: that's ugly use the existing types instead
	public static final long MR = 0;
	public static final long CP = 1;
	
	
	/**
	 * meta data
	 */
	private long rowsInBlock = UNDEFINED;
	private long colsInBlock = UNDEFINED;
	
	private long rows = UNDEFINED;
	private long cols = UNDEFINED;
	
	private long nnz = UNDEFINED;
	
	private long executionContext = UNDEFINED;
	
	public HopsMetaData() 
	{}
	
	
	public HopsMetaData(Hop operator) {
		if(operator != null) {
			this.operator = operator;
			this.rowsInBlock = operator.getRowsInBlock();
			this.colsInBlock = operator.getColsInBlock();
			this.rows = operator.getDim1();
			this.cols = operator.getDim2();
			this.nnz = operator.getNnz();
		}
	}

	public long getRowsInBlock() {
		return rowsInBlock;
	}

	public void setRowsInBlock(long rowsInBlock) {
		this.rowsInBlock = rowsInBlock;
	}

	public long getColsInBlock() {
		return colsInBlock;
	}

	public void setColsInBlock(long colsInBlock) {
		this.colsInBlock = colsInBlock;
	}

	public long getRows() {
		return rows;
	}

	public void setRows(long rows) {
		this.rows = rows;
	}

	public long getCols() {
		return cols;
	}

	public void setCols(long cols) {
		this.cols = cols;
	}

	public Hop getOperator() {
		return operator;
	}

	public long getNnz() {
		return nnz;
	}

	public void setNnz(long nnz) {
		this.nnz = nnz;
	}
	
	@Override
	public String toString() {
		StringBuilder buffer = new StringBuilder();
		
		buffer.append("Hops: " + this.operator.getClass().getSimpleName());
		buffer.append(", ");
		buffer.append("var: " + this.operator.getName());
		buffer.append(", ");
		buffer.append("ID: " + this.operator.getHopID());
		buffer.append(", ");
		buffer.append("dim: (" + this.getRows() + "x" + this.getCols() + ")");
		buffer.append(", ");
		buffer.append("nnz: " + this.getNnz());
		buffer.append(", ");
		buffer.append("blocksize: (" + this.rowsInBlock + "x" + this.colsInBlock + ")");
		
		return buffer.toString();
	}

	public boolean isReblockRequired() {
		return reblockRequired;
	}

	public void setReblockRequired(boolean reblockRequired) {
		this.reblockRequired = reblockRequired;
	}

	public long getExecutionContext() {
		return executionContext;
	}

	public void setExecutionContext(long executionContext) {
		this.executionContext = executionContext;
	}
}
