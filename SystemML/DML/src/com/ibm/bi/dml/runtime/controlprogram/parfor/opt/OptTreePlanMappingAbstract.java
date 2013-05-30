package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;

public class OptTreePlanMappingAbstract extends OptTreePlanMapping
{
	private DMLProgram _prog;
	private Program _rtprog;
	private HashMap<Long, Object> _id_hlprog;
	private HashMap<Long, Object> _id_rtprog;
	
	public OptTreePlanMappingAbstract( )
	{
		super();
		
		_prog = null;
		_rtprog = null;
		
		_id_hlprog = new HashMap<Long, Object>();
		_id_rtprog = new HashMap<Long, Object>();
	}
	
	public void putRootProgram( DMLProgram prog, Program rtprog )
	{
		_prog = prog;
		_rtprog = rtprog;
	}
	
	public long putHopMapping( Hops hops, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_hlprog.put(id, hops);
		_id_rtprog.put(id, null);
		_id_optnode.put(id, n);	
		
		n.setID(id);
		
		return id;
	}
	
	public long putProgMapping( StatementBlock sb, ProgramBlock pb, OptNode n )
	{
		long id = _idSeq.getNextID();
		
		_id_hlprog.put(id, sb);
		_id_rtprog.put(id, pb);
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
	
	public Hops getMappedHop( long id )
	{
		return (Hops)_id_hlprog.get( id );
	}
	
	public Object[] getMappedProg( long id )
	{
		Object[] ret = new Object[2];
		ret[0] = (StatementBlock)_id_hlprog.get( id );
		ret[1] = (ProgramBlock)_id_rtprog.get( id );
		
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
