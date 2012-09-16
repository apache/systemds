package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.HashMap;

import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;

/**
 * Helper class for mapping nodes of the internal plan representation to statement blocks and 
 * hops / function call statements of a given DML program.
 *
 */
public class OptTreePlanMapping 
{
	protected IDSequence _idSeq;
	protected HashMap<Long, OptNode> _id_optnode;
    
	public OptTreePlanMapping()
	{
		_idSeq = new IDSequence();
		_id_optnode = new HashMap<Long, OptNode>();
	}
	
	/**
	 * 
	 * @param id
	 * @return
	 */
	public OptNode getOptNode( long id )
	{
		return _id_optnode.get(id);
	}
	
	/**
	 * 
	 * @param id
	 * @return
	 */
	public long getMappedParentID( long id )
	{
		for( OptNode p : _id_optnode.values() )
			if( p.getChilds() != null )
				for( OptNode c2 : p.getChilds() )
					if( id == c2.getID() )
						return p.getID();
		return -1;
	}
	
	/**
	 * 
	 */
	public void clear()
	{
		_id_optnode.clear();
	}
	
}
