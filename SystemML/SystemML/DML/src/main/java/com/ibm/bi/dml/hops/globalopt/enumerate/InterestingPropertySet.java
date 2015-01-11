/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.InterestingPropertyType;

/**
 * This set characterizes an operator output; for details, see InterestingProperty.
 * Note that each InterestingPropertyType occurs at most once in this set. 
 * 
 */
public class InterestingPropertySet 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected HashMap<InterestingPropertyType, InterestingProperty> _ips = null;
	
	
	public InterestingPropertySet()
	{
		_ips = new HashMap<InterestingPropertyType, InterestingProperty>();
	}
	
	public void addProperties(Collection<InterestingProperty> properties) 
	{
		for(InterestingProperty p : properties)
			_ips.put(p.getType(), p);
	}
	
	public void addProperty(InterestingProperty property) 
	{
		_ips.put(property.getType(), property);
	}
	
	public Collection<InterestingProperty> getProperties() 
	{
		return _ips.values();
	}

	public InterestingProperty getPropertyByType(InterestingPropertyType type) 
	{
		return _ips.get(type);
	}
	
	@Override 
	public boolean equals( Object that )
	{
		//type check for later explicit cast
		if( !(that instanceof InterestingPropertySet ) )
			return false;
		
		InterestingPropertySet thatips = (InterestingPropertySet)that;
		boolean ret = _ips.size() == thatips._ips.size();
		for( Entry<InterestingPropertyType,InterestingProperty> entry : _ips.entrySet() )
		{
			InterestingProperty p1 = entry.getValue();
			InterestingProperty p2 = thatips.getPropertyByType(entry.getKey());		
			ret &= p1.equals(p2);
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
		buffer.append("IPS [");
		for( InterestingProperty p : _ips.values() ) 
			buffer.append( p + " " );
		buffer.append("]");
		return buffer.toString();
	}
}
