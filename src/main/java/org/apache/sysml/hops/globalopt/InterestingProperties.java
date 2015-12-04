/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.hops.globalopt;

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
