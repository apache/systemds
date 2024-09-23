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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.parser.ParForStatementBlock.ResultVar;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDHandler;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.io.FileFormatProperties;
import org.apache.sysds.runtime.io.ListWriter;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageParser;
import static org.apache.sysds.utils.Explain.explain;

import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.LocalFileUtils;
import org.apache.sysds.runtime.util.ProgramConverter;

import scala.Tuple2;

/**
 * Common functionalities for parfor workers in MR jobs. Used by worker wrappers in
 * mappers (base RemoteParFor) and reducers (fused data partitioning and parfor)
 * 
 */
public class RemoteParForUtils 
{
	/**
	 * For remote Spark parfor workers. This is a simplified version compared to MR.
	 * 
	 * @param workerID worker id
	 * @param vars local variable map
	 * @param resultVars list of result variables
	 * @return list of result variables
	 * @throws IOException if IOException occurs
	 */
	public static ArrayList<String> exportResultVariables( long workerID, LocalVariableMap vars, ArrayList<ResultVar> resultVars) 
		throws IOException
	{
		ArrayList<String> ret = new ArrayList<>();
		
		//foreach result variables probe if export necessary
		for( ResultVar rvar : resultVars ) {
			Data dat = vars.get( rvar._name );
			
			if ( dat != null && dat.getDataType().isMatrixOrFrame() ) {
				CacheableData<?> cd = (CacheableData<?>) dat;
				//export result var (iff actually modified in parfor)
				if( cd.isDirty() ) {
					cd.exportData();
					//pass output vars to result (only if actually exported)
					ret.add( ProgramConverter.serializeDataObject(rvar._name, dat) );
				}
				//cleanup pinned result variable from buffer pool
				cd.freeEvictedBlob();
			}
			else if (dat instanceof ListObject) {
				String fname = OptimizerUtils.getUniqueTempFileName();
				ListWriter.writeListToHDFS((ListObject) dat, fname, "binary",
					new FileFormatProperties(ConfigurationManager.getBlocksize()));
				ret.add( ProgramConverter.serializeDataObject(rvar._name, dat) );
			}
		}
		
		return ret;
	}
	
	/**
	 * Export lineage for remote Spark parfor workers.
	 *
	 * @param workerID worker id
	 * @param vars local variable map
	 * @param resultVars list of result variables
	 * @param lineage lineage object
	 * @throws IOException if IOException occurs
	 */
	public static void exportLineageItems(long workerID, LocalVariableMap vars, ArrayList<ResultVar> resultVars, Lineage lineage) 
		throws IOException 
	{
		for( ResultVar rvar : resultVars ) {
			Data dat = vars.get( rvar._name );
			//Note: written lineage should be consistent with exportResultVariables
			if ( dat != null && dat.getDataType() == DataType.MATRIX )  {
				MatrixObject mo = (MatrixObject) dat;
				if( mo.isDirty() ) {
					LineageItem item = lineage.get(rvar._name);
					HDFSTool.writeStringToHDFS(explain(item), mo.getFileName()+".lin");
				}
			}
		}
	}
	
	/**
	 * Cleanup all temporary files created by this SystemDS process.
	 */
	public static void cleanupWorkingDirectories() {
		//delete cache files
		CacheableData.cleanupCacheDir();
		//disable caching (prevent dynamic eviction)
		CacheableData.disableCaching();
		//cleanup working dir (e.g., of CP_FILE instructions)
		LocalFileUtils.cleanupWorkingDirectory();
	}

	/**
	 * Cleanup all temporary files created by this SystemDS process,
	 * on shutdown via exit or interrupt.
	 */
	public static void cleanupWorkingDirectoriesOnShutdown() {
		Runtime.getRuntime().addShutdownHook(
				new DeleteWorkingDirectoriesTask());
	}
	
	public static Lineage[] getLineages(LocalVariableMap[] results) {
		Lineage[] ret = new Lineage[results.length];
		try {
			for( int i=0; i<results.length; i++ ) { //for all workers
				LocalVariableMap vars = results[i];
				ret[i] = new Lineage();
				for( Entry<String,Data> e : vars.entrySet() ) { //for all result vars
					MatrixObject mo = (MatrixObject) e.getValue();
					String lineage = HDFSTool.readStringFromHDFSFile(mo.getFileName()+".lin");
					ret[i].set(e.getKey(),
						LineageParser.parseLineageTrace(lineage, e.getKey()));
				}
			}
		}
		catch(IOException ex) {
			throw new DMLRuntimeException(ex);
		}
		return ret;
	}
	
	
	public static LocalVariableMap[] getResults(List<Tuple2<Long,String>> out, Log LOG )
	{
		HashMap<Long,LocalVariableMap> tmp = new HashMap<>();
		
		int countAll = 0;
		for( Tuple2<Long,String> entry : out ) {
			Long key = entry._1();
			String val = entry._2();
			if( !tmp.containsKey( key ) )
				tmp.put(key, new LocalVariableMap());
			Object[] dat = ProgramConverter.parseDataObject( val );
			tmp.get(key).put((String)dat[0], (Data)dat[1]);
			countAll++;
		}
		
		if( LOG != null ) {
			LOG.debug("Num remote worker results (before deduplication): "+countAll);
			LOG.debug("Num remote worker results: "+tmp.size());
		}
		
		//create return array
		return tmp.values().toArray(new LocalVariableMap[0]);
	}

	/**
	 * Init and register-cleanup of buffer pool
	 * @param workerID worker id
	 * @param isLocal in local spark mode (single JVM)
	 * @throws IOException exception
	 */
	public static void setupBufferPool(long workerID, boolean isLocal) throws IOException {
		//init and register-cleanup of buffer pool (in spark, multiple tasks might
		//share the process-local, i.e., per executor, buffer pool; hence we synchronize
		//the initialization and immediately register the created directory for cleanup
		//on process exit, i.e., executor exit, including any files created in the future.
		synchronized(CacheableData.class) {
			if (!CacheableData.isCachingActive() && !isLocal) {
				//create id, executor working dir, and cache dir
				String uuid = IDHandler.createDistributedUniqueID();
				LocalFileUtils.createWorkingDirectoryWithUUID(uuid);
				CacheableData.initCaching(uuid); //incl activation and cache dir creation
				CacheableData.cacheEvictionLocalFilePrefix = CacheableData.cacheEvictionLocalFilePrefix + "_" + workerID;
				//register entire working dir for delete on shutdown
				RemoteParForUtils.cleanupWorkingDirectoriesOnShutdown();
			}
		}
	}
	
	/**
	 * Task to be registered as shutdown hook in order to delete the 
	 * all working directories, including any remaining files, which 
	 * might not have been created  at time of registration.
	 */
	private static class DeleteWorkingDirectoriesTask extends Thread {
		@Override
		public void run() {
			cleanupWorkingDirectories();
		}
	}
}
