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

import java.io.IOException;

import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.util.HDFSTool;

/**
 * This class serves as the writer for federated objects. The workers are tasked to write their part locally, while
 * the CP writes a mtd file locally containing the addresses and ranges of the workers, enabling `read()` initialization
 * of federated objects.
 *
 * This method, in comparison to other workers, also directly writes the MTD file, this is because it is important
 * that the mtd file is written *AFTER* the workers are finished writing, because their local paths depend on their
 * local configuration. They write into their specified tmp directory.
 */
public class WriterFederated {
	private static final Log LOG = LogFactory.getLog(WriterFederated.class.getName());
	private static final IDSequence siteUniqueCounter = new IDSequence(true);

	/**
	 * Write the federated partitions on the workers and create a mtd file locally to be used to re-read the federate
	 * object.
	 *
	 * @param file                 The file to save to, (defaults to HDFS paths)
	 * @param cd                   The federated object to save.
	 * @param outputFormat         The output format of the file
	 * @param fileFormatProperties The file format properties
	 */
	public static void write(String file, CacheableData<?> cd, String outputFormat, FileFormatProperties fileFormatProperties) {
		LOG.debug("Writing federated map to " + file);
		try {
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path(file);

			long id = cd.getFedMapping().getID();
			FederationMap newFedMap = cd.getFedMapping().mapParallel(id, (range, data) -> {
				String siteFilename = Long.toString(id) + '_' + siteUniqueCounter.getNextID() + '_' + path.getName();
				try {
					FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
						data.getVarID(), new WriteAtSiteUDF(data.getVarID(), siteFilename, outputFormat, fileFormatProperties))).get();
					data.setFilepath((String) response.getData()[0]);
				} catch (Exception e) {
					throw new DMLRuntimeException(e);
				}
				return null;
			});
			// updated filepath
			cd.setFedMapping(newFedMap);

			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
			if (cd instanceof MatrixObject) {
				HDFSTool.writeMetaDataFile(file + ".mtd", cd.getValueType(), cd.getDataCharacteristics(),
						Types.FileFormat.safeValueOf(outputFormat), cd.getPrivacyConstraint(), cd.getFedMapping());
			} else if (cd instanceof FrameObject) {
				HDFSTool.writeMetaDataFile(file + ".mtd", null, ((FrameObject) cd).getSchema(),
						cd.getDataType(), cd.getDataCharacteristics(), Types.FileFormat.safeValueOf(outputFormat),
						cd.getPrivacyConstraint(), cd.getFedMapping());
			} else {
				throw new DMLRuntimeException("TensorObject not yet supported by " + WriterFederated.class.getSimpleName());
			}
		}
		catch(IOException e) {
			throw new DMLRuntimeException("Unable to write test federated matrix to (" + file + "): " + e.getMessage());
		}
	}

	private static class WriteAtSiteUDF extends FederatedUDF {
		private static final long serialVersionUID = -6645546954618784216L;

		private final String _filename;
		private final String _outputFormat;
		private final FileFormatProperties _fileFormatProperties;

		public WriteAtSiteUDF(long input, String filename, String outputFormat, FileFormatProperties fileFormatProperties) {
			super(new long[] {input});
			_filename = filename;
			_outputFormat = outputFormat;
			_fileFormatProperties = fileFormatProperties;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			CacheableData<?> cd = (CacheableData<?>) data[0];
			// Write the file to the local tmp
			String tmpDir = ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.LOCAL_TMP_DIR);
			Path p = new Path(tmpDir + '/' + _filename);
			cd.exportData(p.toString(), _outputFormat, _fileFormatProperties);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, p.toString());
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
