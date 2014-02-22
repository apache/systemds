/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public class OptTreePlanMappingAbstract extends OptTreePlanMapping
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private DMLProgram _prog;
	private Program _rtprog;
	private HashMap<Long, Object> _id_hlprog;
	private HashMap<Long, Object> _id_rtprog;
	private HashMap<Long, Object> _id_symb; // mapping for symbol table
	
	public OptTreePlanMappingAbstract( )
	{
		super();
		
		_prog = null;
		_rtprog = null;
		
		_id_hlprog = new HashMap<Long, Object>();
		_id_rtprog = new HashMap<Long, Object>();
		_id_symb = new HashMap<Long, Object>();
	}
	
	public void putRootProgram( DMLProgram prog, Program rtprog )
	{
		_prog = prog;
		_rtprog = rtprog;
	}
	
	public long putHopMapping( Hop hops, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_hlprog.put(id, hops);
		_id_rtprog.put(id, null);
		_id_symb.put(id, null);
		_id_optnode.put(id, n);	
		
		n.setID(id);
		
		return id;
	}
	
	public long putProgMapping( StatementBlock sb, ProgramBlock pb, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_hlprog.put(id, sb);
		_id_rtprog.put(id, pb);
		_id_symb.put(id, null);
		_id_optnode.put(id, n);
		n.setID(id);
		
		return id;
	}
	
	public Object[] getRootProgram()
	{
		Object[] ret = new Object[2];
		ret[0] = _prog;
		ret[1] = _rtprog;
		return ret;
	}
	
	public Hop getMappedHop( long id )
	{
		return (Hop)_id_hlprog.get( id );
	}
	
	public Object[] getMappedProg( long id )
	{
		Object[] ret = new Object[3];
		ret[0] = (StatementBlock)_id_hlprog.get( id );
		ret[1] = (ProgramBlock)_id_rtprog.get( id );
		ret[2] = (ProgramBlock)_id_symb.get( id );
		
		return ret;
	}
	
	public void replaceMapping( ProgramBlock pb, OptNode n )
	{
		long id = n.getID();
		_id_rtprog.put(id, pb);
		_id_optnode.put(id, n);
	}
	
	@Override
	public void clear()
	{
		super.clear();
		_prog = null;
		_rtprog = null;
		_id_hlprog.clear();
		_id_rtprog.clear();
	}
}
