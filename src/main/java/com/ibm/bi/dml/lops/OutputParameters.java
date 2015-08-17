/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops;

import com.ibm.bi.dml.hops.HopsException;

/**
 * class to maintain output parameters for a lop.
 * 
 */

public class OutputParameters 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum Format {
		TEXT, BINARY, MM, CSV
	};

	private boolean _blocked = true;
	private long _num_rows = -1;
	private long _num_cols = -1;
	private long _nnz = -1;	
	private long _num_rows_in_block = -1;
	private long _num_cols_in_block = -1;
	private String _file_name = null;
	private String _file_label = null;

	Format matrix_format = Format.BINARY;
	
	public String getFile_name() {
		return _file_name;
	}

	public void setFile_name(String fileName) {
		_file_name = fileName;
	}
	
	public String getLabel() {
		return _file_label;
	}

	public void setLabel(String label) {
		_file_label = label;
	}

	public void setDimensions(long rows, long cols, long rows_per_block, long cols_per_block, long nnz) throws HopsException {
		_num_rows = rows;
		_num_cols = cols;
		_nnz = nnz;
		_num_rows_in_block = rows_per_block;
		_num_cols_in_block = cols_per_block;

		if ( _num_rows_in_block == 0 && _num_cols_in_block == 0 ) {
			_blocked = false;
		}
		else if (_num_rows_in_block == -1 && _num_cols_in_block == -1) {
			_blocked = false;
 		}
		else if ( _num_rows_in_block > 0 && _num_cols_in_block > 0 ) {
			_blocked = true;
		}
		else {
			throw new HopsException("In OutputParameters Lop, Invalid values for blocking dimensions: [" + _num_rows_in_block + "," + _num_cols_in_block +"].");
		}
	}

	public Format getFormat() {
		return matrix_format;
	}

	public void setFormat(Format fmt) {
		matrix_format = fmt;
	}

	public boolean isBlocked() {
		return _blocked;
	}

	public void setBlocked(boolean blocked)
	{
		_blocked = blocked;
	}
	
	public long getNumRows()
	{
		return _num_rows;
	}
	
	public void setNumRows(long rows)
	{
		_num_rows = rows;
	}
	
	public long getNumCols()
	{
		return _num_cols;
	}
	
	public void setNumCols(long cols)
	{
		_num_cols = cols;
	}
	
	public Long getNnz() {
		return _nnz;
	}
	
	public void setNnz(long nnz)
	{
		_nnz = nnz;
	}

	public long getRowsInBlock() {
		return _num_rows_in_block;
	}
	
	public void setRowsInBlock(long rows_in_block) {
		_num_rows_in_block = rows_in_block;
	}

	public long getColsInBlock() {
		return _num_cols_in_block;
	}

	public void setColsInBlock(long cols_in_block) {
		_num_cols_in_block = cols_in_block;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("rows=" + getNumRows() + Lop.VALUETYPE_PREFIX);
		sb.append("cols=" + getNumCols() + Lop.VALUETYPE_PREFIX);
		sb.append("nnz=" + getNnz() + Lop.VALUETYPE_PREFIX);
		sb.append("rowsInBlock=" + getRowsInBlock() + Lop.VALUETYPE_PREFIX);
		sb.append("colsInBlock=" + getColsInBlock() + Lop.VALUETYPE_PREFIX);
		sb.append("isBlockedRepresentation=" + isBlocked() + Lop.VALUETYPE_PREFIX);
		sb.append("format=" + getFormat() + Lop.VALUETYPE_PREFIX);
		sb.append("label=" + getLabel() + Lop.VALUETYPE_PREFIX);
		sb.append("filename=" + getFile_name());
		return sb.toString();
	}
}
