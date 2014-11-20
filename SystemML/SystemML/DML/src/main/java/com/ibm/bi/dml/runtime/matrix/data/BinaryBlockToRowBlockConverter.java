/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.matrix.data;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.runtime.util.UtilFunctions;

public class BinaryBlockToRowBlockConverter implements Converter<MatrixIndexes, MatrixBlock, MatrixIndexes, MatrixBlock>
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private MatrixBlock _srcBlock = null;	
	private int _srcRlen = -1;
	private int _srcBrlen = -1;
	private int _pos = -1;
	
	private long _startRowID = -1;
	private long _colID = -1;
	
	//output blocks for reuse
	private MatrixIndexes _destIx = null;
	private MatrixBlock _destBlock = null;
	private Pair<MatrixIndexes, MatrixBlock> _pair = null;
	
	public BinaryBlockToRowBlockConverter()
	{
		_destIx = new MatrixIndexes();
		_destBlock = new MatrixBlock();
		_pair=new Pair<MatrixIndexes, MatrixBlock>(_destIx, _destBlock);
	}
	
	private void reset()
	{
		_srcBlock = null;
		_pos = -1;
		
		_startRowID =-1;
		_colID = -1;
	}
	
	@Override
	public void convert(MatrixIndexes k1, MatrixBlock v1) {
		reset();
		_startRowID = UtilFunctions.cellIndexCalculation(k1.getRowIndex(), _srcBrlen, 0);
		_colID = k1.getColumnIndex();
		_destBlock.reset(1, v1.getNumColumns());
		_srcRlen = v1.getNumRows();
		
		_srcBlock = v1;
		_pos = 0;
	}

	@Override
	public boolean hasNext() 
	{
		//reached end of block
		if( _pos >= _srcRlen )
			return false;
		
		return true;
	}

	@Override
	public Pair<MatrixIndexes, MatrixBlock> next() 
	{
		//check for valid next call
		if( _pos >= _srcRlen )
			return null;
		
		//prepare partial rowblock key
		_destIx.setIndexes(_startRowID+_pos, _colID);
		
		//slice next row
		try {
			//rowlow, rowup, collow, colup (1-based specification)
			_srcBlock.sliceOperations( _pos+1, _pos+1, 1, _srcBlock.getNumColumns(), _destBlock );
		}
		catch(DMLException ex)
		{
			throw new RuntimeException("Failed to slice matrix block into row blocks.", ex);
		}
			
		//increment current pos
		_pos++;
		
		return _pair;
	}

	@Override
	public void setBlockSize(int nr, int nc) {
		_srcBrlen = nr;
	}
}
