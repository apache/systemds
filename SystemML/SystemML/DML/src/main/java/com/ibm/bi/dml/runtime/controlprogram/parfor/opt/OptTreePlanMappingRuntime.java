/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.instructions.Instruction;

public class OptTreePlanMappingRuntime extends OptTreePlanMapping
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<Long, Object> _id_rtprog;

	public OptTreePlanMappingRuntime()
	{
		super();
		_id_rtprog = new HashMap<Long, Object>();
	}
	
	public long putMapping( Instruction inst, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_rtprog.put(id, inst);
		_id_optnode.put(id, n);			
		n.setID(id);
		
		return id;
	}
	
	public long putMapping( ProgramBlock pb, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_rtprog.put(id, pb);
		_id_optnode.put(id, n);
		n.setID(id);
		
		return id;
	}
	
	public void replaceMapping( ProgramBlock pb, OptNode n )
	{
		long id = n.getID();
		_id_rtprog.put(id, pb);
		_id_optnode.put(id, n);
	}
	
	public Object getMappedObject( long id )
	{
		return _id_rtprog.get( id );
	}
	
	public OptNode getOptNode( Object prog )
	{
		for( Entry<Long,Object> e : _id_rtprog.entrySet() )
			if( e.getValue() == prog )
				return _id_optnode.get(e.getKey());
		return null;
	}
	
	@Override
	public void clear()
	{
		super.clear();
		_id_rtprog.clear();
	}
}
