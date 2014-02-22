/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.cost;

/**
 * 
 * 
 */
public class VarStats 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	long _rlen = -1;
	long _clen = -1;
	long _brlen = -1;
	long _bclen = -1;
	double _nnz = -1;
	boolean _inmem = false;
	
	public VarStats( long rlen, long clen, long brlen, long bclen, long nnz, boolean inmem )
	{
		_rlen = rlen;
		_clen = clen;
		_brlen = brlen;
		_bclen = bclen;
		_nnz = nnz;
		_inmem = inmem;
	}
	
	/**
	 * 
	 * @return
	 */
	public double getSparsity()
	{
		return (_nnz<0) ? 1.0 : (double)_nnz/_rlen/_clen;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("VarStats: [");
		sb.append("rlen = ");
		sb.append(_rlen);
		sb.append(", clen = ");
		sb.append(_clen);
		sb.append(", nnz = ");
		sb.append(_nnz);
		sb.append(", inmem = ");
		sb.append(_inmem);
		sb.append("]");
	
		return sb.toString();
	}
	
	@Override
	public Object clone()
	{
		return new VarStats(_rlen, _clen, _brlen, _bclen,(long)_nnz, _inmem );
	}
}
