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

package org.apache.sysds.runtime.io;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;

public class ListWriter
{
	/**
	 * Writes a list object and all contained objects to a folder with related meta data.
	 * The individual objects (including nested lists) are written with existing matrix/frame 
	 * writers and meta data such that the entire list and separate objects can be read back.
	 * By using the existing writers, all formats are naturally supported and we can ensure
	 * consistency of the on-disk representation.
	 * 
	 * @param lo list object
	 * @param fname directory name 
	 * @param fmtStr format string
	 * @param props file format properties
	 * @throws DMLRuntimeException if write fails
	 */
	public static void writeListToHDFS(ListObject lo, String fname, String fmtStr, FileFormatProperties props)
		throws DMLRuntimeException
	{
		DataCharacteristics dc = new MatrixCharacteristics(lo.getLength(), 1, 0, 0);
		
		try {
			//write basic list meta data
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			HDFSTool.writeMetaDataFile(fname + ".mtd", lo.getValueType(), null,
				lo.getDataType(), dc, FileFormat.safeValueOf(fmtStr), props);
			
			//create folder for list w/ appropriate permissions
			HDFSTool.createDirIfNotExistOnHDFS(fname,
				DMLConfig.DEFAULT_SHARED_DIR_PERMISSION);
			
			//write regular/named list by position/position_name
			//TODO additional parallelization over objects (in addition to parallel writers)
			for(int i=0; i<lo.getLength(); i++) {
				Data dat = lo.getData(i);
				String lfname = fname +"/"+i+"_"+(lo.isNamedList()?lo.getName(i):"null");
				if( dat instanceof CacheableData<?> ) {
					((CacheableData<?>)dat).exportData(lfname, fmtStr, props);
					IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(job, new Path(lfname));
				}
				else if( dat instanceof ListObject )
					writeListToHDFS((ListObject)dat, lfname, fmtStr, props);
				else //scalar
					HDFSTool.writeScalarToHDFS((ScalarObject)dat, lfname);
			}
			
			//remove crc file of list directory
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(job, new Path(fname));
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(
				"Failed to write list object of length "+dc.getRows()+".", ex);
		}
	}
}
