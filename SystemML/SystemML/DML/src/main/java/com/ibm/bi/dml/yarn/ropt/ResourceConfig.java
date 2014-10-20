/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import java.util.ArrayList;
import java.util.Collections;

import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;

public class ResourceConfig 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private double _cpres = -1;
	private ArrayList<Double> _mrres = null;
	
	public ResourceConfig( double cp, ArrayList<Double> mr )
	{
		_cpres = cp;
		_mrres = mr;
	}
	
	public ResourceConfig( ArrayList<ProgramBlock> prog, double init )
	{
		//init cp memory
		_cpres = init;
		
		//init mr memory
		addProgramBlocks(prog, init);
	}
	
	/**
	 * 
	 * @return
	 */
	public long getCPResource()
	{
		return (long)_cpres;
	}
	
	public void setCPResource( long res )
	{
		_cpres = res;
	}
	
	public void setMRResources( ArrayList<ProgramBlock> B, double[][] res ) 
		throws DMLRuntimeException
	{
		if( _mrres.size() != res.length )
			throw new DMLRuntimeException("Memo table sizes do not match: "+_mrres.size()+" vs "+res.length);
		
		int len = res.length;
		for( int i=0; i<len; i++ )
			_mrres.set(i, res[i][0]);
	}
	
	/**
	 * 
	 * @return
	 */
	public long getMaxMRResource()
	{
		double val = Collections.max(_mrres);
		return (long)val;
	}
	
	/**
	 * 
	 * @return
	 */
	public String serialize() 
	{
		StringBuilder ret = new StringBuilder();
		
		//serialize cp
		ret.append("cp");
		ret.append("-");
		ret.append(_cpres);
		ret.append(":");
		
		//serialize mr
		int len = _mrres.size();
		for( int i=0; i<len; i++ ) {
			ret.append(i);
			ret.append("-");
			ret.append(_mrres.get(i).toString());
			ret.append(":");
		}
		
		return ret.toString();
	}
	
	public static ResourceConfig deserialize( String str ) 
	{	
		String[] pairs = str.split(":");
		
		//deserialize cp
		double cp = Double.valueOf(pairs[0].substring(3));
		
		//deserialize mr
		ArrayList<Double> mr = new ArrayList<Double>();
		for (int i=1; i<pairs.length; i++) 
		{
			if (pairs[i].length() > 0) {
				String[] entries = pairs[i].split("_");
				int index = Integer.parseInt(entries[0]);
				double val = Double.parseDouble(entries[1]);
				mr.set(index, val);
			}
		}
		
		return new ResourceConfig(cp, mr);
	}
	
	/**
	 * 
	 * @param pbs
	 * @param init
	 */
	private void addProgramBlocks( ArrayList<ProgramBlock> pbs, double init )
	{
		for( ProgramBlock pb : pbs )
			addProgramBlock(pb, init);
	}
	
	/**
	 * 
	 * @param pb
	 * @param init
	 */
	private void addProgramBlock( ProgramBlock pb, double init )
	{
		if (pb instanceof FunctionProgramBlock)
		{
			FunctionProgramBlock fpb = (FunctionProgramBlock)pb;
			addProgramBlocks(fpb.getChildBlocks(), init);
		}
		else if (pb instanceof WhileProgramBlock)
		{
			//TODO while predicate 
			WhileProgramBlock fpb = (WhileProgramBlock)pb;
			addProgramBlocks(fpb.getChildBlocks(), init);
		}	
		else if (pb instanceof IfProgramBlock)
		{
			//TODO if predicate 
			IfProgramBlock fpb = (IfProgramBlock)pb;
			addProgramBlocks(fpb.getChildBlocksIfBody(), init);
			addProgramBlocks(fpb.getChildBlocksElseBody(), init);
		}
		else if (pb instanceof ForProgramBlock) //incl parfor
		{
			//TODO while predicate 
			ForProgramBlock fpb = (ForProgramBlock)pb;
			addProgramBlocks(fpb.getChildBlocks(), init);
		}
		else
		{
			//for objects hash is unique because memory location used
			_mrres.add(init);
		}
	}
}
