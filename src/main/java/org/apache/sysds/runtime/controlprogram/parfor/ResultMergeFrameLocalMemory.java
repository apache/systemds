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

package org.apache.sysds.runtime.controlprogram.parfor;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.util.UtilFunctions;

public class ResultMergeFrameLocalMemory extends ResultMerge<FrameObject>
{
	private static final long serialVersionUID = 549739254879310540L;
	
	public ResultMergeFrameLocalMemory(FrameObject out, FrameObject[] in, String outputFilename, boolean accum) {
		super( out, in, outputFilename, accum );
	}
	
	@Override
	public FrameObject executeSerialMerge() 
	{
		FrameObject foNew = null; //always create new matrix object (required for nested parallelism)
		
		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (local, in-memory): Execute serial merge for output "
				+_output.hashCode()+" (fname="+_output.getFileName()+")");
		
		try
		{
			//get old and new output frame blocks
			FrameBlock outFB = _output.acquireRead();
			FrameBlock outFBNew = new FrameBlock(outFB);
			
			//create compare matrix if required (existing data in result)
			FrameBlock compare = outFB;
			int rlen = compare.getNumRows();
			int clen = compare.getNumColumns();
			
			//serial merge all inputs
			boolean flagMerged = false;
			for( FrameObject in : _inputs )
			{
				//check for empty inputs (no iterations executed)
				if( in != null && in != _output ) 
				{
					if( LOG.isTraceEnabled() )
						LOG.trace("ResultMergeFrame (local, in-memory): Merge input "+in.hashCode()+" (fname="+in.getFileName()+")");
					
					//read/pin input_i
					FrameBlock inMB = in.acquireRead();
					
					//core merge
					for(int j=0; j<clen; j++) {
						ValueType vt = compare.getSchema()[j];
						for(int i=0; i<rlen; i++) {
							Object val1 = compare.get(i, j);
							Object val2 = inMB.get(i, j);
							if( UtilFunctions.compareTo(vt, val1, val2) != 0 )
								outFBNew.set(i, j, val2);
						}
					}
					
					//unpin and clear in-memory input_i
					in.release();
					in.clearData();
					flagMerged = true;
				}
			}
			
			//create output and release old output
			foNew =  flagMerged ? createNewFrameObject(_output, outFBNew) : _output;
			_output.release();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}

		//LOG.trace("ResultMerge (local, in-memory): Executed serial merge for output "+_output.getVarName()+" (fname="+_output.getFileName()+") in "+time.stop()+"ms");
		
		return foNew;
	}
	
	@Override
	public FrameObject executeParallelMerge( int par ) {
		if( LOG.isTraceEnabled() )
			LOG.trace("ResultMerge (local, in-memory): Execute parallel (par="+par+") "
				+ "merge for output "+_output.hashCode()+" (fname="+_output.getFileName()+")");
		return executeSerialMerge();
	}

	private static FrameObject createNewFrameObject( FrameObject foOld, FrameBlock dataNew ) {
		FrameObject ret = new FrameObject(foOld);
		ret.acquireModify(dataNew);
		ret.release();
		return ret;
	}
}
