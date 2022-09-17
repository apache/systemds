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

import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.io.WriterCompressed;

public class MatrixWriterFactory
{

	public static MatrixWriter createMatrixWriter(FileFormat fmt) {
		return createMatrixWriter(fmt, -1, null);
	}

	public static MatrixWriter createMatrixWriter(FileFormat fmt, int replication, FileFormatProperties props)
	{
		MatrixWriter writer = null;

		switch(fmt) {
			case TEXT:
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					writer = new WriterTextCellParallel();
				else
					writer = new WriterTextCell();
				break;

			case MM:
				//note: disabled parallel cp write of matrix market in order to ensure the
				//requirement of writing out a single file

				//if( OptimizerUtils.PARALLEL_CP_WRITE_TEXTFORMATS )
				//	writer = new WriterMatrixMarketParallel();
				writer = new WriterMatrixMarket();
				break;

			case CSV:
				if( props!=null && !(props instanceof FileFormatPropertiesCSV) )
					throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					writer = new WriterTextCSVParallel((FileFormatPropertiesCSV)props);
				else
					writer = new WriterTextCSV((FileFormatPropertiesCSV)props);
				break;

			case LIBSVM:
				if(props != null && !(props instanceof FileFormatPropertiesLIBSVM))
					throw new DMLRuntimeException("Wrong type of file format properties for LIBSVM writer.");
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					writer = new WriterTextLIBSVMParallel((FileFormatPropertiesLIBSVM) props);
				else
					writer = new WriterTextLIBSVM((FileFormatPropertiesLIBSVM) props);
				break;

			case BINARY:
				if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS) )
					writer = new WriterBinaryBlockParallel(replication);
				else
					writer = new WriterBinaryBlock(replication);
				break;

			case HDF5:
				if(props != null && !(props instanceof FileFormatPropertiesHDF5))
					throw new DMLRuntimeException("Wrong type of file format properties for HDF5 writer.");
				else if( ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS) )
					return new WriterHDF5Parallel((FileFormatPropertiesHDF5) props);
				else
					return new WriterHDF5((FileFormatPropertiesHDF5) props);

			case COMPRESSED:
				return WriterCompressed.create(props);

			default:
				throw new DMLRuntimeException("Failed to create matrix writer for unknown format: " + fmt.toString());
		}

		return writer;
	}
}
