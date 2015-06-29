/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.lops.compile.RecompileStatus;
import com.ibm.bi.dml.parser.Expression.DataType;
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
	
	public void init( ArrayList<Hop> hops, RecompileStatus status)
	{
		//check existing status
		if(    hops == null ||  hops.isEmpty() || status == null 
			|| status.getTWriteStats().isEmpty() )
		{
			return; //nothing to do
		}
		
		//population via recursive search for treads
		Hop.resetVisitStatus(hops);
		for( Hop hop : hops )
			rinit(hop, status);
	}
	
	/**
	 * 
	 * @param hops
	 * @param status
	 */
	public void extract( ArrayList<Hop> hops, RecompileStatus status)
	{
		//check existing status
		if( status == null )
			return; //nothing to do
		
		//clear old cached state
		status.clearStatus();
		
		//extract all transient writes (must be dag root)
		for( Hop hop : hops ) {
			if(    hop instanceof DataOp 
				&& ((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTWRITE )
			{
				String varname = hop.getName();
				Hop input = hop.getInput().get(0); //child
				MatrixCharacteristics mc = getAllInputStats(input);
				if( mc != null )
					status.getTWriteStats().put(varname, mc);
			}
		}
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
					dim1 = (dim1<=0) ? tmp.getRows() : dim1;
					dim2 = (dim2<=0) ? tmp.getCols() : dim2;
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
				dim1 = (dim1<=0) ? tmp.getRows() : dim1;
				dim2 = (dim2<=0) ? tmp.getCols() : dim2;
				nnz = (nnz<=0) ? tmp.getNonZeros() : nnz;
			}
			ret = new MatrixCharacteristics(dim1, dim2, -1, -1, nnz);
		}
		
		return ret;
	}
	
	public boolean hasInputStatistics(Hop h) 
	{
		boolean ret = false;
		
		//determine if any input has useful exact/worst-case stats
		for( Hop in : h.getInput() )
			if( in.dimsKnown() || _memo.containsKey(in.getHopID()) ) 
			{
				ret = true;
				break;
			}
		
		//determine if hop itself has worst-case stats (this is important
		//for transient read with cross-dag worst-case estimates)
		if( h instanceof DataOp && ((DataOp)h).getDataOpType()==DataOpTypes.TRANSIENTREAD ){
			return true;
		}
			
		return ret;
	}
	
	/**
	 * 
	 * @param hop
	 * @param status
	 */
	private void rinit(Hop hop, RecompileStatus status) 
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//probe status of previous twrites
		if(    hop instanceof DataOp && hop.getDataType() == DataType.MATRIX
			&& ((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTREAD )
		{
			String varname = hop.getName();
			MatrixCharacteristics mc = status.getTWriteStats().get(varname);
			if( mc != null )
				_memo.put(hop.getHopID(), mc);
		}
		
		if( hop.getInput()!=null && !hop.getInput().isEmpty() )
			for( Hop c : hop.getInput() )
				rinit( c, status );
			
		hop.setVisited(VisitStatus.DONE);
	}	
	
}
