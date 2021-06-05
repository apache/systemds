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

import java.util.Arrays;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.HDFSTool;

public class ListReader 
{
	/**
	 * Reads a list object and all contained objects from a folder with related meta data.
	 * The individual objects (including nested lists) are read with existing matrix/frame 
	 * readers and meta data such that the entire list and separate objects can be restored.
	 * By using the existing readers, all formats are naturally supported and we can ensure
	 * consistency of the on-disk representation.
	 * 
	 * @param fname directory name 
	 * @param fmtStr format string
	 * @param props file format properties
	 * @return list object
	 * @throws DMLRuntimeException if inconsistent meta data or read fails
	 */
	public static ListObject readListFromHDFS(String fname, String fmtStr, FileFormatProperties props)
		throws DMLRuntimeException
	{
		MetaDataAll meta = new MetaDataAll(fname+".mtd", false, true);
		int numObjs = (int) meta.getDim1();
		boolean named = false;
		
		Data[] data = null;
		String[] names = null;
		try {
			// read all meta data files
			Path dirPath = new Path(fname);
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			FileSystem fs = IOUtilFunctions.getFileSystem(dirPath, job);
			Path[] mtdFiles = IOUtilFunctions.getMetadataFilePaths(fs, dirPath);
			if( numObjs != mtdFiles.length ) {
				throw new DMLRuntimeException("List meta data does not match "
					+ "available mtd files: "+numObjs+" vs "+mtdFiles.length);
			}
			
			// determine if regular or named list
			named = Arrays.stream(mtdFiles).map(p -> p.toString())
				.anyMatch(s -> !s.substring(s.lastIndexOf('_')).equals("null"));
			data = new Data[numObjs];
			names = named ? new String[numObjs] : null;
			
			// read all individual files (but only create objects for 
			// matrices and frames, which are then read on demand via acquire())
			for( int i=0; i<numObjs; i++ ) {
				MetaDataAll lmeta = new MetaDataAll(mtdFiles[i].toString(), false, true);
				String lfname = lmeta.getFilename().substring(0, lmeta.getFilename().length()-4);
				DataCharacteristics dc = lmeta.getDataCharacteristics();
				FileFormat fmt = lmeta.getFileFormat();
				Data dat = null;
				switch( lmeta.getDataType() ) {
					case MATRIX:
						dat = new MatrixObject(lmeta.getValueType(), lfname);
						break;
					case TENSOR:
						dat = new TensorObject(lmeta.getValueType(), lfname);
						break;
					case FRAME:
						dat = new FrameObject(lfname);
						if( lmeta.getSchema() != null )
							((FrameObject)dat).setSchema(lmeta.getSchema());
						break;
					case LIST:
						dat = ListReader.readListFromHDFS(lfname, fmt.toString(), props);
						break;
					case SCALAR:
						dat = HDFSTool.readScalarObjectFromHDFSFile(lfname, lmeta.getValueType());
						break;
					default:
						throw new DMLRuntimeException("Unexpected data type: " + lmeta.getDataType());
				}
				
				if(dat instanceof CacheableData<?>) {
					((CacheableData<?>)dat).setMetaData(new MetaDataFormat(dc, fmt));
					((CacheableData<?>)dat).enableCleanup(false); // disable delete
				}

				String[] parts = lfname.substring(lfname.lastIndexOf("/")+1).split("_");
				data[Integer.parseInt(parts[0])] = dat;
				if( named )
					names[Integer.parseInt(parts[0])] = parts[1];
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(
				"Failed to write list object of length "+numObjs+".", ex);
		}
		
		// construct list object
		return named ? new ListObject(data, names) : new ListObject(data);
	}
}
