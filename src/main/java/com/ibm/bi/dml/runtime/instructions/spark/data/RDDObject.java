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

package com.ibm.bi.dml.runtime.instructions.spark.data;

import org.apache.spark.api.java.JavaPairRDD;

public class RDDObject extends LineageObject
{

	private JavaPairRDD<?,?> _rddHandle = null;
	
	//meta data on origin of given rdd handle
	private boolean _checkpointed = false; //created via checkpoint instruction
	private boolean _hdfsfile = false;     //created from hdfs file
	
	public RDDObject( JavaPairRDD<?,?> rddvar, String varName)
	{
		_rddHandle = rddvar;
		_varName = varName;
	}
	
	/**
	 * 
	 * @return
	 */
	public JavaPairRDD<?,?> getRDD()
	{
		return _rddHandle;
	}
	
	public void setCheckpointRDD( boolean flag )
	{
		_checkpointed = flag;
	}
	
	public boolean isCheckpointRDD() 
	{
		return _checkpointed;
	}
	
	public void setHDFSFile( boolean flag )
	{
		_hdfsfile = flag;
	}
	
	public boolean isHDFSFile()
	{
		return _hdfsfile;
	}
	

	/**
	 * 
	 * @return
	 */
	public boolean allowsShortCircuitRead()
	{
		boolean ret = false;
		
		if( isCheckpointRDD() && getLineageChilds().size() == 1 ) {
			LineageObject lo = getLineageChilds().get(0);
			ret = ( lo instanceof RDDObject && ((RDDObject)lo).isHDFSFile() );
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean allowsShortCircuitCollect()
	{
		return ( isCheckpointRDD() && getLineageChilds().size() == 1
			     && getLineageChilds().get(0) instanceof RDDObject );
	}
}
