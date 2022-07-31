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

import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.InetSocketAddress;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.controlprogram.parfor.util.IDSequence;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.meta.DataCharacteristics;

/**
 * This class serves as the reader for federated objects. To read the files a mdt file is required. The reader is
 * different from the other readers in the since that it does not return a MatrixBlock but a Matrix Object wrapper,
 * containing the federated Mapping.
 * 
 * On the Matrix Object the function isFederated() will if called read in the federated locations and instantiate the
 * map. The reading is done through this code.
 * 
 * This means in practice that it circumvent the other reading code. See more in:
 * 
 * org.apache.sysds.runtime.controlprogram.caching.MatrixObject.readBlobFromHDFS()
 * org.apache.sysds.runtime.controlprogram.caching.CacheableData.isFederated()
 * 
 */
public class ReaderWriterFederated {
	private static final Log LOG = LogFactory.getLog(ReaderWriterFederated.class.getName());
	private static final IDSequence siteUniqueCounter = new IDSequence(true);

	/**
	 * Read a federated map from disk, It is not initialized before it is used in:
	 * 
	 * org.apache.sysds.runtime.instructions.fed.InitFEDInstruction
	 * 
	 * @param file The file to read (defaults to HDFS)
	 * @param mc   The data characteristics of the file, that can be read from the mtd file.
	 * @return A List of federatedRanges and Federated Data
	 */
	public static List<Pair<FederatedRange, FederatedData>> read(String file, DataCharacteristics mc) {
		LOG.debug("Reading federated map from " + file);
		try {
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path(file);
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			FSDataInputStream data = fs.open(path);
			ObjectMapper mapper = new ObjectMapper();
			List<FederatedDataAddress> obj = mapper.readValue(data, new TypeReference<List<FederatedDataAddress>>() {
			});
			return obj.stream().map(x -> x.convert()).collect(Collectors.toList());
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Unable to read federated matrix (" + file + ")", e);
		}
	}

	/**
	 * TODO add writing to each of the federated locations so that they also save their matrices.
	 * 
	 * Currently this would write the federated matrix to disk only locally.
	 * 
	 * @param file   The file to save to, (defaults to HDFS paths)
	 * @param fedMap The federated map to save.
	 */
	public static void write(String file, FederationMap fedMap) {
		LOG.debug("Writing federated map to " + file);
		try {
			JobConf job = new JobConf(ConfigurationManager.getCachedJobConf());
			Path path = new Path(file);
			FileSystem fs = IOUtilFunctions.getFileSystem(path, job);
			
			fedMap.forEachParallel((range, data) -> {
				String siteFilename = Long.toString(siteUniqueCounter.getNextID()) + '_' + path.getName();
				try {
					FederatedResponse response = data.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
						data.getVarID(), new WriteAtSiteUDF(data.getVarID(), siteFilename))).get();
					data.setFilepath((String) response.getData()[0]);
				} catch (Exception e) {
					throw new DMLRuntimeException(e);
				}
				return null;
			});
			
			DataOutputStream out = fs.create(path, true);
			ObjectMapper mapper = new ObjectMapper();
			FederatedDataAddress[] outObjects = parseMap(fedMap.getMap());
			try(BufferedWriter pw = new BufferedWriter(new OutputStreamWriter(out))) {
				mapper.writeValue(pw, outObjects);
			}

			IOUtilFunctions.deleteCrcFilesFromLocalFileSystem(fs, path);
		}
		catch(IOException e) {
			throw new DMLRuntimeException("Unable to write test federated matrix to (" + file + "): " + e.getMessage());
		}
	}

	private static FederatedDataAddress[] parseMap(List<Pair<FederatedRange, FederatedData>> map) {
		return map.stream()
			.map(e -> new FederatedDataAddress(e.getKey(), e.getValue()))
			.toArray(FederatedDataAddress[]::new);
	}

	/**
	 * This class is used for easy serialization from json using Jackson. The warnings are suppressed because the
	 * setters and getters only is used inside Jackson.
	 */
	@SuppressWarnings("unused")
	private static class FederatedDataAddress {
		private Types.DataType _dataType;
		private InetSocketAddress _address;
		private String _filepath;
		private long[] _begin;
		private long[] _end;

		public FederatedDataAddress() {
		}

		protected FederatedDataAddress(FederatedRange fr, FederatedData fd) {
			_dataType = fd.getDataType();
			_address = fd.getAddress();
			_filepath = fd.getFilepath();
			_begin = fr.getBeginDims();
			_end = fr.getEndDims();
		}

		protected Pair<FederatedRange, FederatedData> convert() {
			FederatedRange fr = new FederatedRange(_begin, _end);
			FederatedData fd = new FederatedData(_dataType, _address, _filepath);
			return new ImmutablePair<>(fr, fd);
		}

		public String getFilepath() {
			return _filepath;
		}

		public void setFilepath(String filePath) {
			_filepath = filePath;
		}

		public Types.DataType getDataType() {
			return _dataType;
		}

		public void setDataType(Types.DataType dataType) {
			_dataType = dataType;
		}

		public InetSocketAddress getAddress() {
			return _address;
		}

		public void setAddress(InetSocketAddress address) {
			_address = address;
		}

		public long[] getBegin() {
			return _begin;
		}

		public void setBegin(long[] begin) {
			_begin = begin;
		}

		public long[] getEnd() {
			return _end;
		}

		public void setEnd(long[] end) {
			_end = end;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(_dataType);
			sb.append(" ");
			sb.append(_address);
			sb.append(" ");
			sb.append(_filepath);
			sb.append(" ");
			sb.append(Arrays.toString(_begin));
			sb.append(" ");
			sb.append(Arrays.toString(_end));
			return sb.toString();
		}
	}

	private static class WriteAtSiteUDF extends FederatedUDF {
		private static final long serialVersionUID = -6645546954618784216L;

		private final String _filename;

		public WriteAtSiteUDF(long input, String filename) {
			super(new long[] {input});
			_filename = filename;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject mo = (MatrixObject) data[0];
			String tmpDir = ConfigurationManager.getDMLConfig().getTextValue(DMLConfig.LOCAL_TMP_DIR);
			Path p = new Path(tmpDir + '/' + _filename);
			mo.exportData(p.toString(), null);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, p.toString());
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
