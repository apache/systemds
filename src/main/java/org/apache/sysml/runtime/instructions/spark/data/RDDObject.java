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

package org.apache.sysml.runtime.instructions.spark.data;

import org.apache.spark.api.java.JavaPairRDD;

public class RDDObject extends LineageObject
{

	private JavaPairRDD<?,?> _rddHandle = null;
	
	//meta data on origin of given rdd handle
	private boolean _checkpointed = false; //created via checkpoint instruction
	private boolean _hdfsfile = false;     //created from hdfs file
	private String  _hdfsFname = null;     //hdfs filename, if created from hdfs.  
	
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
	
	public void setHDFSFile( boolean flag ) {
		_hdfsfile = flag;
	}
	
	public void setHDFSFilename( String fname ) {
		_hdfsFname = fname;
	}
	
	public boolean isHDFSFile() {
		return _hdfsfile;
	}
	
	public String getHDFSFilename() {
		return _hdfsFname;
	}
	

	/**
	 * Indicates if rdd is an hdfs file or a checkpoint over an hdfs file;
	 * in both cases, we can directly read the file instead of collecting
	 * the given rdd.
	 * 
	 * @return
	 */
	public boolean allowsShortCircuitRead()
	{
		boolean ret = isHDFSFile();
		
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
	
	/**
	 * 
	 * @return
	 */
	public boolean rHasCheckpointRDDChilds()
	{
		//probe for checkpoint rdd
		if( _checkpointed )
			return true;
		
		//process childs recursively
		boolean ret = false;
		for( LineageObject lo : getLineageChilds() ) {
			if( lo instanceof RDDObject )
				ret |= ((RDDObject)lo).rHasCheckpointRDDChilds();
		}
		
		return ret;
	}
}
