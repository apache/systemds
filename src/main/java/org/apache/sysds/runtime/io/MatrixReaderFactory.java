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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.compress.io.ReaderCompressed;
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public class MatrixReaderFactory {
	private static final Log LOG = LogFactory.getLog(MatrixReaderFactory.class.getName());
	public static MatrixReader createMatrixReader(FileFormat fmt) {
		MatrixReader reader = null;
		boolean par = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS);
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;

		if (LOG.isDebugEnabled()){
			LOG.debug("reading parallel: " + par + " mcsr: " + mcsr);
		}

		switch(fmt) {
			case TEXT:
			case MM:
				reader = (par & mcsr) ?
					new ReaderTextCellParallel(fmt) : new ReaderTextCell(fmt);
				break;

			case CSV:
				reader = (par & mcsr) ?
					new ReaderTextCSVParallel(new FileFormatPropertiesCSV()) :
					new ReaderTextCSV(new FileFormatPropertiesCSV());
				break;

			case LIBSVM:
				reader = (par & mcsr) ? new ReaderTextLIBSVMParallel(
					new FileFormatPropertiesLIBSVM()) : new ReaderTextLIBSVM(new FileFormatPropertiesLIBSVM());
				break;

			case BINARY:
				reader = (par & mcsr) ?
					new ReaderBinaryBlockParallel(false) : new ReaderBinaryBlock(false);
				break;

			case HDF5:
				reader = (par & mcsr) ? 
					new ReaderHDF5Parallel(new FileFormatPropertiesHDF5()) : 
					new ReaderHDF5(new FileFormatPropertiesHDF5());
				break;

			case COG:
				reader = (par & mcsr) ?
					new ReaderCOGParallel(new FileFormatPropertiesCOG()) :
					new ReaderCOG(new FileFormatPropertiesCOG());
				break;

			case COMPRESSED:
				reader = ReaderCompressed.create();
				break;
			
			default:
				throw new DMLRuntimeException("Failed to create matrix reader for unknown format: " + fmt.toString());
		}
		return reader;
	}

	public static MatrixReader createMatrixReader( ReadProperties props )  {
		//check valid read properties
		if( props == null )
			throw new DMLRuntimeException("Failed to create matrix reader with empty properties.");

		MatrixReader reader = null;
		FileFormat fmt = props.fmt;
		boolean par = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS);
		boolean mcsr = MatrixBlock.DEFAULT_SPARSEBLOCK == SparseBlock.Type.MCSR;

		if (LOG.isDebugEnabled()){
			LOG.debug("reading parallel: " + par + " mcsr: " + mcsr);
		}

		switch(fmt) {
			case TEXT:
			case MM:
				reader = (par & mcsr) ?
					new ReaderTextCellParallel(fmt) : new ReaderTextCell(fmt);
				break;

			case CSV:
				reader = (par & mcsr) ?
					new ReaderTextCSVParallel( props.formatProperties!=null ?
						(FileFormatPropertiesCSV)props.formatProperties : new FileFormatPropertiesCSV()) :
					new ReaderTextCSV( props.formatProperties!=null ?
						(FileFormatPropertiesCSV)props.formatProperties : new FileFormatPropertiesCSV());
				break;

			case LIBSVM:
				FileFormatPropertiesLIBSVM fileFormatPropertiesLIBSVM = props.formatProperties != null ? (FileFormatPropertiesLIBSVM) props.formatProperties : new FileFormatPropertiesLIBSVM();
				reader = (par & mcsr) ? new ReaderTextLIBSVMParallel(fileFormatPropertiesLIBSVM) : new ReaderTextLIBSVM(
					fileFormatPropertiesLIBSVM);
				break;

			case BINARY:
				reader = (par & mcsr) ?
					new ReaderBinaryBlockParallel(props.localFS) : new ReaderBinaryBlock(props.localFS);
				break;

			case HDF5:
				FileFormatPropertiesHDF5 fileFormatPropertiesHDF5 = props.formatProperties != null ? (FileFormatPropertiesHDF5) props.formatProperties : new FileFormatPropertiesHDF5();
				reader = (par & mcsr) ? new ReaderHDF5Parallel(fileFormatPropertiesHDF5) : new ReaderHDF5(
					fileFormatPropertiesHDF5);
				break;

			case COG:
				FileFormatPropertiesCOG fileFormatPropertiesCOG = props.formatProperties != null ? (FileFormatPropertiesCOG) props.formatProperties : new FileFormatPropertiesCOG();
				reader = (par & mcsr) ?
						new ReaderCOGParallel(fileFormatPropertiesCOG) : new ReaderCOG(fileFormatPropertiesCOG);
				break;

			case COMPRESSED:
				reader = new ReaderCompressed();
				break;
			default:
				throw new DMLRuntimeException("Failed to create matrix reader for unknown format: " + fmt.toString());
		}
		return reader;
	}
}
