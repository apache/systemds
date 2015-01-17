/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;

/**
 * Memoization Table (hop id, worst-case matrix characteristics).
 * 
 */
public class MemoTable 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private HashMap<Long, MatrixCharacteristics> _memo = null;
	
	public MemoTable()
	{
		_memo = new HashMap<Long, MatrixCharacteristics>();
	}

	/**
	 * 
	 * @param hopID
	 * @param dim1
	 * @param dim2
	 * @param nnz
	 */
	public void memoizeStatistics( long hopID, long dim1, long dim2, long nnz )
	{
		_memo.put(hopID, new MatrixCharacteristics(dim1, dim2, -1, -1, nnz));
	}
		
	/**
	 * 
	 * @param inputs
	 * @return
	 */
	public MatrixCharacteristics[] getAllInputStats( ArrayList<Hop> inputs )
	{
		MatrixCharacteristics[] ret = new MatrixCharacteristics[inputs.size()];
		for( int i=0; i<inputs.size(); i++ )
		{
			Hop input = inputs.get(i);
			
			long dim1 = input.getDim1();
			long dim2 = input.getDim2();
			long nnz = input.getNnz();
			
			if( input.dimsKnown() ) //all dims known
			{
				ret[i] = new MatrixCharacteristics(dim1, dim2, -1, -1, nnz);
			}
			else
			{
				MatrixCharacteristics tmp = _memo.get(input.getHopID());
				if( tmp != null )
				{
					//enrich exact information with worst-case stats
					dim1 = (dim1<=0) ? tmp.get_rows() : dim1;
					dim2 = (dim2<=0) ? tmp.get_cols() : dim2;
					nnz = (nnz<=0) ? tmp.getNonZeros() : nnz;
				}
				ret[i] = new MatrixCharacteristics(dim1, dim2, -1, -1, nnz);
			}
		}
		
		return ret;
	}

	public MatrixCharacteristics getAllInputStats( Hop input )
	{
		MatrixCharacteristics ret = null;
			
		long dim1 = input.getDim1();
		long dim2 = input.getDim2();
		long nnz = input.getNnz();
		
		if( input.dimsKnown() ) //all dims known
		{
			ret = new MatrixCharacteristics(dim1, dim2, -1, -1, nnz);
		}
		else
		{
			MatrixCharacteristics tmp = _memo.get(input.getHopID());
			if( tmp != null )
			{
				//enrich exact information with worst-case stats
				dim1 = (dim1<=0) ? tmp.get_rows() : dim1;
				dim2 = (dim2<=0) ? tmp.get_cols() : dim2;
				nnz = (nnz<=0) ? tmp.getNonZeros() : nnz;
			}
			ret = new MatrixCharacteristics(dim1, dim2, -1, -1, nnz);
		}
		
		return ret;
	}
	
	public boolean hasInputStatistics(Hop h) 
	{
		boolean ret = false;
		for( Hop in : h.getInput() )
			if( in.dimsKnown() || _memo.containsKey(in.getHopID()) ) 
			{
				ret = true;
				break;
			}
		
		return ret;
	}
}
