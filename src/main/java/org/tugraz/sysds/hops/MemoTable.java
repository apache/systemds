/*
 * Modifications Copyright 2019 Graz University of Technology
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.tugraz.sysds.hops;

import org.tugraz.sysds.common.Types.DataType;
import org.tugraz.sysds.hops.Hop.DataOpTypes;
import org.tugraz.sysds.hops.recompile.RecompileStatus;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Memoization Table (hop id, worst-case matrix characteristics).
 * 
 */
public class MemoTable 
{
	
	private HashMap<Long, DataCharacteristics> _memo = null;
	
	public MemoTable()
	{
		_memo = new HashMap<>();
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

	public void init( Hop hop, RecompileStatus status)
	{
		//check existing status
		if(    hop == null || status == null 
			|| status.getTWriteStats().isEmpty() )
		{
			return; //nothing to do
		}
		
		//population via recursive search for treads
		hop.resetVisitStatus();
		rinit(hop, status);
	}

	public void extract( ArrayList<Hop> hops, RecompileStatus status)
	{
		//check existing status
		if( status == null )
			return; //nothing to do
		
		//extract all transient writes (must be dag root)
		for( Hop hop : hops ) {
			if(    hop instanceof DataOp 
				&& ((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTWRITE )
			{
				String varname = hop.getName();
				Hop input = hop.getInput().get(0); //child
				DataCharacteristics dc = getAllInputStats(input);
				if( dc != null )
					status.getTWriteStats().put(varname, dc);
				else
					status.getTWriteStats().remove(varname);
			}
		}
	}

	public void memoizeStatistics( long hopID, long dim1, long dim2, long nnz ) {
		_memo.put(hopID, new MatrixCharacteristics(dim1, dim2, -1, nnz));
	}

	public DataCharacteristics[] getAllInputStats(ArrayList<Hop> inputs )
	{
		if( inputs == null )
			return null;
		
		DataCharacteristics[] ret = new MatrixCharacteristics[inputs.size()];
		for( int i=0; i<inputs.size(); i++ )
		{
			Hop input = inputs.get(i);
			
			long dim1 = input.getDim1();
			long dim2 = input.getDim2();
			long nnz = input.getNnz();
			
			if( input.dimsKnown() ) //all dims known
			{
				ret[i] = new MatrixCharacteristics(dim1, dim2, -1, nnz);
			}
			else
			{
				DataCharacteristics tmp = _memo.get(input.getHopID());
				if( tmp != null )
				{
					//enrich exact information with worst-case stats
					dim1 = (dim1<=0) ? tmp.getRows() : dim1;
					dim2 = (dim2<=0) ? tmp.getCols() : dim2;
					nnz = (nnz<=0) ? tmp.getNonZeros() : nnz;
				}
				ret[i] = new MatrixCharacteristics(dim1, dim2, -1, nnz);
			}
		}
		
		return ret;
	}

	public DataCharacteristics getAllInputStats(Hop input )
	{
		if( input == null )
			return null;
		
		DataCharacteristics ret = null;
			
		long dim1 = input.getDim1();
		long dim2 = input.getDim2();
		long nnz = input.getNnz();
		
		if( input.dimsKnown(true) ) //all dims known
		{
			ret = new MatrixCharacteristics(dim1, dim2, -1, nnz);
		}
		else //enrich exact information with worst-case stats
		{
			DataCharacteristics tmp = _memo.get(input.getHopID());
			if( tmp != null ) {
				dim1 = (dim1<=0) ? tmp.getRows() : dim1;
				dim2 = (dim2<=0) ? tmp.getCols() : dim2;
				nnz = (nnz<0) ? tmp.getNonZeros() : nnz;
			}
			ret = new MatrixCharacteristics(dim1, dim2, -1, nnz);
		}
		
		return ret;
	}

	public boolean hasInputStatistics(Hop h) 
	{
		boolean ret = false;
		
		//determine if any input has useful exact/worst-case stats
		for( Hop in : h.getInput() )
			if( in.dimsKnownAny() || _memo.containsKey(in.getHopID()) ) {
				ret = true;
				break;
			}
		
		//determine if hop itself has worst-case stats (this is important
		//for transient read with cross-dag worst-case estimates)
		if(   (h instanceof DataOp && ((DataOp)h).getDataOpType()==DataOpTypes.TRANSIENTREAD)
		    ||(h instanceof DataGenOp) ) 
		{
			ret = true;
		}
		
		return ret;
	}

	private void rinit(Hop hop, RecompileStatus status) 
	{
		if( hop.isVisited() )
			return;
		
		//probe status of previous twrites
		if(    hop instanceof DataOp && hop.getDataType() == DataType.MATRIX
			&& ((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTREAD )
		{
			String varname = hop.getName();
			DataCharacteristics dc = status.getTWriteStats().get(varname);
			if( dc != null )
				_memo.put(hop.getHopID(), dc);
		}
		
		if( hop.getInput()!=null && !hop.getInput().isEmpty() )
			for( Hop c : hop.getInput() )
				rinit( c, status );
			
		hop.setVisited();
	}	
	
}
