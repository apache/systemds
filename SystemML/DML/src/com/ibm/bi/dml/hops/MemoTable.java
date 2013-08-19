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
	public MatrixCharacteristics[] getAllInputStats( ArrayList<Hops> inputs )
	{
		MatrixCharacteristics[] ret = new MatrixCharacteristics[inputs.size()];
		for( int i=0; i<inputs.size(); i++ )
		{
			Hops input = inputs.get(i);
			
			long dim1 = input.get_dim1();
			long dim2 = input.get_dim2();
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

	public MatrixCharacteristics getAllInputStats( Hops input )
	{
		MatrixCharacteristics ret = null;
			
		long dim1 = input.get_dim1();
		long dim2 = input.get_dim2();
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
	
	public boolean hasInputStatistics(Hops h) 
	{
		boolean ret = false;
		for( Hops in : h.getInput() )
			if( in.dimsKnown() || _memo.containsKey(in.getHopID()) ) 
			{
				ret = true;
				break;
			}
		
		return ret;
	}
}
