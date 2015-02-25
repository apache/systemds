/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.parser.DataExpression;

public class CSVFileFormatProperties extends FileFormatProperties 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private boolean header;
	private String delim;
	private boolean fill;
	private double fillValue;
	
	private boolean sparse;
	
	public CSVFileFormatProperties() {
		super(FileFormat.CSV);
		
		// get the default values for CSV properties from the language layer
		this.header = DataExpression.DEFAULT_DELIM_HAS_HEADER_ROW;
		this.delim = DataExpression.DEFAULT_DELIM_DELIMITER;
		this.fill = DataExpression.DEFAULT_DELIM_FILL;
		this.fillValue = DataExpression.DEFAULT_DELIM_FILL_VALUE;
		this.sparse = DataExpression.DEFAULT_DELIM_SPARSE;
	}
	
	public CSVFileFormatProperties(boolean hasHeader, String delim, boolean fill, double fillValue) {
		super(FileFormat.CSV);
		
		this.header = hasHeader;
		this.delim = delim;
		this.fill = fill;
		this.fillValue = fillValue;
	}

	public CSVFileFormatProperties(boolean hasHeader, String delim, boolean sparse) {
		super(FileFormat.CSV);
		
		this.header = hasHeader;
		this.delim = delim;
		this.sparse = sparse;
	}

	public boolean hasHeader() {
		return header;
	}

	public void setHeader(boolean hasHeader) {
		this.header = hasHeader;
	}

	public String getDelim() {
		return delim;
	}

	public void setDelim(String delim) {
		this.delim = delim;
	}

	public boolean isFill() {
		return fill;
	}

	public void setFill(boolean fill) {
		this.fill = fill;
	}

	public double getFillValue() {
		return fillValue;
	}

	public void setFillValue(double fillValue) {
		this.fillValue = fillValue;
	}

	public boolean isSparse() {
		return sparse;
	}

	public void setSparse(boolean sparse) {
		this.sparse = sparse;
	}
	
}
