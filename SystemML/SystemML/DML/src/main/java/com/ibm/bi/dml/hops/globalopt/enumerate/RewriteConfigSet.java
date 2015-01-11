/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;


public class RewriteConfigSet 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<RewriteConfigType, RewriteConfig> _rcs = null;
	
	public RewriteConfigSet()
	{
		_rcs = new HashMap<RewriteConfigType, RewriteConfig>();
	}
	
	public RewriteConfigSet( RewriteConfigSet that )
	{
		this();
		_rcs.putAll( that._rcs );
	}
	
	public void addConfig(RewriteConfig conf) 
	{
		_rcs.put(conf.getType(), conf);
	}

	public void addConfigs(Collection<RewriteConfig> configs) 
	{
		for(RewriteConfig conf : configs) {
			_rcs.put(conf.getType(), conf);
		}
	}
	
	public Collection<RewriteConfig> getConfigs() 
	{
		return _rcs.values();
	}

	public RewriteConfig getConfigByType(RewriteConfigType type)
	{
		return _rcs.get(type);
	}
	

	@Override 
	public boolean equals( Object that )
	{
		//type check for later explicit cast
		if( !(that instanceof RewriteConfigSet ) )
			return false;
		
		RewriteConfigSet thatips = (RewriteConfigSet)that;
		boolean ret = _rcs.size() == thatips._rcs.size();
		for( Entry<RewriteConfigType,RewriteConfig> entry : _rcs.entrySet() )
		{
			RewriteConfig c1 = entry.getValue();
			RewriteConfig c2 = thatips.getConfigByType(entry.getKey());		
			ret &= c1.equals(c2);
			if( !ret ) break; //early abort
		}
		return ret;
	}
	
	@Override
	public int hashCode()
	{
		return super.hashCode();
	}
	
	@Override
	public String toString() 
	{
		StringBuilder buffer = new StringBuilder();
		buffer.append("RCS [");
		for( RewriteConfig p : _rcs.values() ) 
			buffer.append( p + " " );
		buffer.append("]");
		return buffer.toString();
	}
	
}
