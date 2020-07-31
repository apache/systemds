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

public class FrameReaderFactory {
	protected static final Log LOG = LogFactory.getLog(FrameReaderFactory.class.getName());

	public static FrameReader createFrameReader(FileFormat fmt) {
		if( LOG.isDebugEnabled() )
			LOG.debug("Creating Frame Reader " + fmt);
		FileFormatProperties props = (fmt == FileFormat.CSV) ? new FileFormatPropertiesCSV() : null;
		return createFrameReader(fmt, props);
	}

	public static FrameReader createFrameReader(FileFormat fmt, FileFormatProperties props) {
		if( LOG.isDebugEnabled() )
			LOG.debug("Creating Frame Reader " + fmt + props);
		FrameReader reader = null;

		switch(fmt) {
			case TEXT:
				if(ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS))
					reader = new FrameReaderTextCellParallel();
				else
					reader = new FrameReaderTextCell();
				break;

			case CSV:
				if(props != null && !(props instanceof FileFormatPropertiesCSV))
					throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
				if(ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_TEXTFORMATS))
					reader = new FrameReaderTextCSVParallel((FileFormatPropertiesCSV) props);
				else
					reader = new FrameReaderTextCSV((FileFormatPropertiesCSV) props);
				break;

			case BINARY:
				if(ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_READ_BINARYFORMATS))
					reader = new FrameReaderBinaryBlockParallel();
				else
					reader = new FrameReaderBinaryBlock();
				break;
			case PROTO:
				// TODO performance improvement: add parallel reader
				reader = new FrameReaderProto();
				break;

			default:
				throw new DMLRuntimeException("Failed to create frame reader for unknown format: " + fmt.toString());
		}

		return reader;
	}
}
