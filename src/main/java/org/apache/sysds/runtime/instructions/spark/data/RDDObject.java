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

package org.apache.sysds.runtime.instructions.spark.data;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class RDDObject extends LineageObject
{
	private JavaPairRDD<?,?> _rddHandle = null;
	
	//meta data on origin of given rdd handle
	private boolean _checkpointed = false; //created via checkpoint instruction
	private boolean _hdfsfile = false;     //created from hdfs file
	private String  _hdfsFname = null;     //hdfs filename, if created from hdfs.  
	private boolean _parRDD = false;       //is a parallelized rdd at driver
	private boolean _pending = true;       //is a pending rdd operation
	private DataCharacteristics _dc = null;
	
	public RDDObject( JavaPairRDD<?,?> rddvar) {
		super();
		_rddHandle = rddvar;
	}

	public JavaPairRDD<?,?> getRDD() {
		return _rddHandle;
	}

	public void setRDD(JavaPairRDD<?,?> rddHandle) {
		_rddHandle = rddHandle;
	}
	
	public void setCheckpointRDD( boolean flag ) {
		_checkpointed = flag;
	}
	
	public boolean isCheckpointRDD() {
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
	
	public void setParallelizedRDD( boolean flag ) {
		_parRDD = flag;
	}
	
	public boolean isParallelizedRDD() {
		return _parRDD; 
	}
	
	public void setPending(boolean flag) {
		_pending = flag;
	}
	
	public boolean isPending() {
		return _pending;
	}

	public void setDataCharacteristics(DataCharacteristics dc) {
		_dc = dc;
	}

	public DataCharacteristics getDataCharacteristics() {
		return _dc;
	}
	

	/**
	 * Indicates if rdd is an hdfs file or a checkpoint over an hdfs file;
	 * in both cases, we can directly read the file instead of collecting
	 * the given rdd.
	 * 
	 * @return true if rdd is an hdfs file or a checkpoint over an hdfs file
	 */
	public boolean allowsShortCircuitRead()
	{
		// Cannot trust the hdfs file for reused RDD objects
		if (isInLineageCache() && isCheckpointRDD())
			return false;

		boolean ret = isHDFSFile();
		
		if( isCheckpointRDD() && getLineageChilds().size() == 1 ) {
			LineageObject lo = getLineageChilds().get(0);
			ret = ( lo instanceof RDDObject && ((RDDObject)lo).isHDFSFile() );
		}
		
		return ret;
	}

	public boolean allowsShortCircuitCollect()
	{
		// If the RDD is marked to be persisted and cached locally, we want to collect the RDD
		// so that the next time we can reuse the RDD.
		if (isInLineageCache())
			return false;

		return ( isCheckpointRDD() && getLineageChilds().size() == 1
			     && getLineageChilds().get(0) instanceof RDDObject );
	}

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
