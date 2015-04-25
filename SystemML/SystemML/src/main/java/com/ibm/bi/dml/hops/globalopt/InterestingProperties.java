/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.google.common.base.Objects;

/**
 * An instance of this class represents one 'interesting property set' defined by the instances
 * of all interesting properties. We do not use objects per interesting property in order to
 * (1) simplify the creation and interesting property sets, and (2) prevent excessive object
 * creation and garbage collection.
 * 
 */
public class InterestingProperties 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public enum Location {
		MEM,
		HDFS_CACHE,
		HDFS,
	}
	
	public enum Format {
		ANY,
		BINARY_BLOCK,
		BINARY_CELL,
		TEXT_CELL,
		TEXT_MM,
		TEXT_CSV,
	}
	
	public enum Partitioning {
		NONE,
		ROW_WISE,
		COL_WISE,
		//ROW_BLOCK_WISE,
		//COL_BLOCK_WISE,
	}
	
	//supported interesting properties
	private int          _blocksize   = -1;    //-1 for any
	private Format       _format      = null;
	private Location     _location    = null;
	private Partitioning _pformat     = null;
	private int          _replication = -1;
	private boolean      _emptyblocks = false;         
	
	
	public InterestingProperties( int blocksize, Format format, Location location, Partitioning pformat, int replication, boolean emptyblocks )
	{
		_blocksize   = blocksize;
		_format      = format;
		_location    = location;
		_pformat     = pformat;
		_replication = replication;
		_emptyblocks = emptyblocks;
	}
	
	public InterestingProperties( InterestingProperties that )
	{
		_blocksize   = that._blocksize;
		_format      = that._format;
		_location    = that._location;
		_pformat     = that._pformat;
		_replication = that._replication;
		_emptyblocks = that._emptyblocks;
	}
	
	@Override
	public boolean equals(Object o)
	{
		if( !(o instanceof InterestingProperties) )
			return false;
		
		InterestingProperties that = (InterestingProperties)o;
		return (    _blocksize   == that._blocksize
				 && _format      == that._format
				 && _location    == that._location
				 && _pformat     == that._pformat
				 && _replication == that._replication
				 && _emptyblocks == that._emptyblocks );
	}
	
	@Override
	public int hashCode()
	{
		//relies on google's guava library 
		return Objects.hashCode(
				   _blocksize, 
				   (_format!=null)?_format.ordinal():-1,
				   (_location!=null)?_location.ordinal():-1,
				   (_pformat!=null)?_pformat.ordinal():-1,
				   _replication,
				   _emptyblocks
			   );
	}
	
	@Override
	public String toString()
	{
		return "IPS[" + _blocksize + "," + _format + "," + _location + "," + _pformat + "," + _replication + "," + _emptyblocks + "]";
	}
}
