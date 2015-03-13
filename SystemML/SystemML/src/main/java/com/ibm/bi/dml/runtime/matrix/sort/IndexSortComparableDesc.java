/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.matrix.sort;

import org.apache.hadoop.io.DoubleWritable;

public class IndexSortComparableDesc extends IndexSortComparable
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Override
	public int compareTo(Object o) 
	{
		//descending order (note: we cannot just inverted the ascending order)
		if( o instanceof DoubleWritable ) {
			int tmp = _dval.compareTo((DoubleWritable) o);
			return (( tmp!=0 ) ? -1*tmp : tmp); //prevent -0
		}
		//compare double value and index (e.g., for stable sort)
		else if( o instanceof IndexSortComparable) {
			IndexSortComparable that = (IndexSortComparable)o;
			int tmp = _dval.compareTo(that._dval);
			tmp = (( tmp!=0 ) ? -1*tmp : tmp); //prevent -0
			if( tmp==0 ) //secondary sort
				tmp = _lval.compareTo(that._lval);
			return tmp;
		}	
		else {
			throw new RuntimeException("Unsupported comparison involving class: "+o.getClass().getName());
		}
		
	}
}
