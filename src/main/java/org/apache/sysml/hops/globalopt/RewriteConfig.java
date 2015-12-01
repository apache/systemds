/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.hops.globalopt;

import org.apache.sysml.hops.Hop.FileFormatTypes;
import org.apache.sysml.hops.globalopt.InterestingProperties.Location;
import org.apache.sysml.lops.LopProperties.ExecType;



/**
 * This RewriteConfig represents an instance configuration of a particular rewrite.
 * 
 */
public class RewriteConfig 
{
	
	/*
	public enum RewriteConfigType {
		BLOCK_SIZE,
		FORMAT_CHANGE,
		EXEC_TYPE,
		DATA_PARTITIONING,
		VECTORIZATION,
		REPLICATION_FACTOR
	}
	*/
	
	private ExecType _rewriteSetExecType  = null;
	private int      _rewriteSetBlockSize = -1;
	private FileFormatTypes _rewriteFormat = null;
	
	public RewriteConfig( ExecType et, int bs, FileFormatTypes format )
	{
		_rewriteSetExecType = et;
		_rewriteSetBlockSize = bs;
		_rewriteFormat = format;
	}
	
	public RewriteConfig( RewriteConfig rc )
	{
		_rewriteSetExecType = rc._rewriteSetExecType;
		_rewriteSetBlockSize = rc._rewriteSetBlockSize;
		_rewriteFormat = rc._rewriteFormat;
	}
	
	public ExecType getExecType()
	{
		return _rewriteSetExecType;
	}
	
	public int getBlockSize()
	{
		return _rewriteSetBlockSize;
	}
	
	public FileFormatTypes getFormat()
	{
		return _rewriteFormat;
	}
	
	/**
	 * 
	 * @return
	 */
	public InterestingProperties deriveInterestingProperties()
	{
		int bs = _rewriteSetBlockSize; 
		Location loc = (_rewriteSetExecType==ExecType.CP) ? Location.MEM : Location.HDFS;
		
		return new InterestingProperties(bs, null, loc, null, -1, true);
	}

	
	@Override
	public String toString() 
	{
		return "RC["+_rewriteSetExecType+","+_rewriteSetBlockSize+"]";//+_type+"="+_value + "]";
	}

}
