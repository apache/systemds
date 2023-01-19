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

public class FrameWriterFactory {
	protected static final Log LOG = LogFactory.getLog(FrameWriterFactory.class.getName());

	public static FrameWriter createFrameWriter(FileFormat fmt) {
		return createFrameWriter(fmt, null);
	}

	public static FrameWriter createFrameWriter(FileFormat fmt, FileFormatProperties props) {
		boolean textParallel = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS);
		boolean binaryParallel = ConfigurationManager.getCompilerConfigFlag(ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS);
		switch(fmt) {
			case TEXT:
				return textParallel ? new FrameWriterTextCellParallel() : new FrameWriterTextCell();
			case CSV:
				if(props != null && !(props instanceof FileFormatPropertiesCSV))
					throw new DMLRuntimeException("Wrong type of file format properties for CSV writer.");
				FileFormatPropertiesCSV fp = (FileFormatPropertiesCSV) props;
				return textParallel ? new FrameWriterTextCSVParallel(fp) : new FrameWriterTextCSV(fp);
			case COMPRESSED:
				return new FrameWriterCompressed(binaryParallel);
			case BINARY:
				return binaryParallel ? new FrameWriterBinaryBlockParallel() : new FrameWriterBinaryBlock();
			case PROTO:
				return new FrameWriterProto();
			default:
				throw new DMLRuntimeException("Failed to create frame writer for unknown format: " + fmt.toString());
		}
	}
}
