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

package org.apache.sysds.runtime.instructions.fed;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.fedplanner.FTypes.FType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedStatistics;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.lineage.LineageTraceable;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;

public class InitFEDInstruction extends FEDInstruction implements LineageTraceable {

	private static final Log LOG = LogFactory.getLog(InitFEDInstruction.class.getName());

	public static final String FED_MATRIX_IDENTIFIER = "matrix";
	public static final String FED_FRAME_IDENTIFIER = "frame";

	private CPOperand _type, _addresses, _ranges, _localObject, _output;

	public InitFEDInstruction(CPOperand type, CPOperand addresses, CPOperand ranges, CPOperand out, String opcode,
		String instr) {
		super(FEDType.Init, opcode, instr);
		_type = type;
		_addresses = addresses;
		_ranges = ranges;
		_output = out;
	}

	public InitFEDInstruction(CPOperand type, CPOperand addresses, CPOperand ranges, CPOperand object, CPOperand out, String opcode,
		String instr) {
		this(type, addresses, ranges, out, opcode, instr);
		_localObject = object;
	}

	public static InitFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// We need 5 parts: Opcode, Type (Frame/Matrix), Addresses (list of Strings with
		// url/ip:port/filepath), ranges and the output Operand
		if(parts.length != 5 && parts.length != 6)
			throw new DMLRuntimeException("Invalid number of operands in federated instruction: " + str);
		String opcode = parts[0];

		if(parts.length == 5) {
			CPOperand type, addresses, ranges, out;
			type = new CPOperand(parts[1]);
			addresses = new CPOperand(parts[2]);
			ranges = new CPOperand(parts[3]);
			out = new CPOperand(parts[4]);
			return new InitFEDInstruction(type, addresses, ranges, out, opcode, str);
		} else {
			CPOperand type, addresses, object, ranges, out;
			type = new CPOperand(parts[1]);
			addresses = new CPOperand(parts[2]);
			ranges = new CPOperand(parts[3]);
			object = new CPOperand(parts[4]);
			out = new CPOperand(parts[5]);
			return new InitFEDInstruction(type, addresses, ranges, object, out, opcode, str);
		}
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if(_localObject == null)
			processFedInit(ec);
		else
			processFromLocalFedInit(ec);
	}

	private void processFedInit(ExecutionContext ec){
		String type = ec.getScalarInput(_type).getStringValue();
		ListObject addresses = ec.getListObject(_addresses.getName());
		ListObject ranges = ec.getListObject(_ranges.getName());
		List<Pair<FederatedRange, FederatedData>> feds = new ArrayList<>();

		if(addresses.getLength() * 2 != ranges.getLength())
			throw new DMLRuntimeException("Federated read needs twice the amount of addresses as ranges " + "(begin and end): addresses=" + addresses.getLength() + " ranges=" + ranges.getLength());

		//check for duplicate addresses (would lead to overwrite with common variable names)
		// TODO relax requirement by using different execution contexts per federated data?
		Set<String> addCheck = new HashSet<>();
		for( Data dat : addresses.getData() )
			if( dat instanceof StringObject ) {
				String address = ((StringObject) dat).getStringValue();
				if(addCheck.contains(address))
					LOG.warn("Federated data contains address duplicates: " + addresses);
				addCheck.add(address);
			}

		Types.DataType fedDataType;
		if(type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER))
			fedDataType = Types.DataType.MATRIX;
		else if(type.equalsIgnoreCase(FED_FRAME_IDENTIFIER))
			fedDataType = Types.DataType.FRAME;
		else
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");

		long[] usedDims = new long[] {0, 0};
		for(int i = 0; i < addresses.getLength(); i++) {
			Data addressData = addresses.getData().get(i);
			if(addressData instanceof StringObject) {
				// We split address into url/ip, the port and file path of file to read
				String[] parsedValues = parseURL(((StringObject) addressData).getStringValue());
				String host = parsedValues[0];
				int port = Integer.parseInt(parsedValues[1]);
				String filePath = parsedValues[2];

				if(DMLScript.FED_STATISTICS)
					// register the federated worker for federated statistics creation
					FederatedStatistics.registerFedWorker(host, port);

				// get beginning and end of data ranges
				List<Data> rangesData = ranges.getData();
				Data beginData = rangesData.get(i * 2);
				Data endData = rangesData.get(i * 2 + 1);
				if(beginData.getDataType() != Types.DataType.LIST || endData.getDataType() != Types.DataType.LIST)
					throw new DMLRuntimeException("Federated read ranges (lower, upper) have to be lists of dimensions");
				List<Data> beginDimsData = ((ListObject) beginData).getData();
				List<Data> endDimsData = ((ListObject) endData).getData();

				// fill begin and end dims
				long[] beginDims = new long[beginDimsData.size()];
				long[] endDims = new long[beginDims.length];
				for(int d = 0; d < beginDims.length; d++) {
					beginDims[d] = ((ScalarObject) beginDimsData.get(d)).getLongValue();
					endDims[d] = ((ScalarObject) endDimsData.get(d)).getLongValue();
				}

				usedDims[0] = Math.max(usedDims[0], endDims[0]);
				usedDims[1] = Math.max(usedDims[1], endDims[1]);
				try {
					FederatedData federatedData = new FederatedData(fedDataType,
						new InetSocketAddress(InetAddress.getByName(host), port), filePath);
					feds.add(new ImmutablePair<>(new FederatedRange(beginDims, endDims), federatedData));
				}
				catch(UnknownHostException e) {
					throw new DMLRuntimeException("federated host was unknown: " + host);
				}
			}
			else {
				throw new DMLRuntimeException("federated instruction only takes strings as addresses");
			}
		}

		if(type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER)) {
			CacheableData<?> output = ec.getCacheableData(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateMatrix(output, feds, null);
		}
		else if(type.equalsIgnoreCase(FED_FRAME_IDENTIFIER)) {
			if(usedDims[1] > Integer.MAX_VALUE)
				throw new DMLRuntimeException("federated Frame can not have more than max int columns, because the "
					+ "schema can only be max int length");
			FrameObject output = ec.getFrameObject(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateFrame(output, feds, null);
		}
		else {
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");
		}
	}

	public void processFromLocalFedInit(ExecutionContext ec) {
		String type = ec.getScalarInput(_type).getStringValue();
		ListObject addresses = ec.getListObject(_addresses.getName());
		ListObject ranges = ec.getListObject(_ranges.getName());
		List<Pair<FederatedRange, FederatedData>> feds = new ArrayList<>();

		CacheableData<?> co = ec.getCacheableData(_localObject);
		CacheBlock<?> cb =  co.acquireReadAndRelease();

		if(addresses.getLength() * 2 != ranges.getLength())
			throw new DMLRuntimeException("Federated read needs twice the amount of addresses as ranges "
				+ "(begin and end): addresses=" + addresses.getLength() + " ranges=" + ranges.getLength());

		//check for duplicate addresses (would lead to overwrite with common variable names)
		Set<String> addCheck = new HashSet<>();
		for(Data dat : addresses.getData())
			if(dat instanceof StringObject) {
				String address = ((StringObject) dat).getStringValue();
				if(addCheck.contains(address))
					LOG.warn("Federated data contains address duplicates: " + addresses);
				addCheck.add(address);
			}

		Types.DataType fedDataType;
		if(type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER))
			fedDataType = Types.DataType.MATRIX;
		else if(type.equalsIgnoreCase(FED_FRAME_IDENTIFIER))
			fedDataType = Types.DataType.FRAME;
		else
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");

		long[] usedDims = new long[] {0, 0};
		CacheBlock<?>[] cbs = new CacheBlock<?>[addresses.getLength()];
		for(int i = 0; i < addresses.getLength(); i++) {
			Data addressData = addresses.getData().get(i);
			if(addressData instanceof StringObject) {
				// We split address into url/ip, the port and file path of file to read
				String[] parsedValues = parseURLNoFilePath(((StringObject) addressData).getStringValue());
				String host = parsedValues[0];
				int port = Integer.parseInt(parsedValues[1]);
				String filePath = co.getFileName();

				if(DMLScript.FED_STATISTICS)
					// register the federated worker for federated statistics creation
					FederatedStatistics.registerFedWorker(host, port);

				// get beginning and end of data ranges
				List<Data> rangesData = ranges.getData();
				Data beginData = rangesData.get(i * 2);
				Data endData = rangesData.get(i * 2 + 1);
				if(beginData.getDataType() != Types.DataType.LIST || endData.getDataType() != Types.DataType.LIST)
					throw new DMLRuntimeException(
						"Federated read ranges (lower, upper) have to be lists of dimensions");
				List<Data> beginDimsData = ((ListObject) beginData).getData();
				List<Data> endDimsData = ((ListObject) endData).getData();

				// fill begin and end dims
				long[] beginDims = new long[beginDimsData.size()];
				long[] endDims = new long[beginDims.length];
				for(int d = 0; d < beginDims.length; d++) {
					beginDims[d] = ((ScalarObject) beginDimsData.get(d)).getLongValue();
					endDims[d] = ((ScalarObject) endDimsData.get(d)).getLongValue();
				}
				usedDims[0] = Math.max(usedDims[0], endDims[0]);
				usedDims[1] = Math.max(usedDims[1], endDims[1]);

				CacheBlock<?> slice = cb instanceof MatrixBlock ? ((MatrixBlock)cb).slice((int) beginDims[0], (int) endDims[0]-1, (int) beginDims[1], (int) endDims[1]-1, true) :
					((FrameBlock)cb).slice((int) beginDims[0], (int) endDims[0]-1, (int) beginDims[1], (int) endDims[1]-1, true, new FrameBlock());
				cbs[i] = slice;

				try {
					FederatedData federatedData = new FederatedData(fedDataType,
						new InetSocketAddress(InetAddress.getByName(host), port), filePath);
					feds.add(new ImmutablePair<>(new FederatedRange(beginDims, endDims), federatedData));
				}
				catch(UnknownHostException e) {
					throw new DMLRuntimeException("federated host was unknown: " + host);
				}
			}
			else {
				throw new DMLRuntimeException("federated instruction only takes strings as addresses");
			}
		}

		if(type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER)) {
			CacheableData<?> output = ec.getCacheableData(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateMatrix(output, feds, cbs);
		}
		else if(type.equalsIgnoreCase(FED_FRAME_IDENTIFIER)) {
			if(usedDims[1] > Integer.MAX_VALUE)
				throw new DMLRuntimeException("federated Frame can not have more than max int columns, because the "
					+ "schema can only be max int length");
			FrameObject output = ec.getFrameObject(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateFrame(output, feds, cbs);
		}
		else {
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");
		}
	}

	public static String[] parseURLNoFilePath(String input) {
		try {
			// Artificially making it http protocol.
			// This is to avoid malformed address error in the URL passing.
			// TODO: Construct new protocol name for Federated communication
			URL address = new URL("http://" + input);
			String host = address.getHost();
			if(host.length() == 0)
				throw new IllegalArgumentException("Missing Host name for federated address");
			// The current system does not support ipv6, only ipv4.
			// TODO: Support IPV6 address for Federated communication
			String ipRegex = "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$";
			if(host.matches("^\\d+\\.\\d+\\.\\d+\\.\\d+$") && !host.matches(ipRegex))
				throw new IllegalArgumentException("Input Host address looks like an IP address but is outside range");
			int port = address.getPort();
			if(port == -1)
				port = DMLConfig.DEFAULT_FEDERATED_PORT;
			if(address.getQuery() != null)
				throw new IllegalArgumentException("Query is not supported");

			if(address.getRef() != null)
				throw new IllegalArgumentException("Reference is not supported");

			return new String[] {host, String.valueOf(port)};
		}
		catch(MalformedURLException e) {
			throw new IllegalArgumentException(
				"federated address `" + input + "` does not fit required URL pattern of \"host:port/directory\"", e);
		}
	}

	public static String[] parseURL(String input) {
		try {
			// Artificially making it http protocol.
			// This is to avoid malformed address error in the URL passing.
			// TODO: Construct new protocol name for Federated communication
			URL address = new URL("http://" + input);
			String host = address.getHost();
			if(host.length() == 0)
				throw new IllegalArgumentException("Missing Host name for federated address");
			// The current system does not support ipv6, only ipv4.
			// TODO: Support IPV6 address for Federated communication
			String ipRegex = "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$";
			if(host.matches("^\\d+\\.\\d+\\.\\d+\\.\\d+$") && !host.matches(ipRegex))
				throw new IllegalArgumentException("Input Host address looks like an IP address but is outside range");
			int port = address.getPort();
			if(port == -1)
				port = DMLConfig.DEFAULT_FEDERATED_PORT;
			String filePath = address.getPath();
			if(filePath.length() <= 1)
				throw new IllegalArgumentException("Missing File path for federated address");
			// Remove the first character making the path Dynamic from the location of the worker.
			// This is in contrast to before where it was static paths
			filePath = filePath.substring(1);
			// To make static file paths use double "//" EG:
			// example.dom//staticFile.txt
			// example.dom/dynamicFile.txt
			if(address.getQuery() != null)
				throw new IllegalArgumentException("Query is not supported");

			if(address.getRef() != null)
				throw new IllegalArgumentException("Reference is not supported");

			return new String[] {host, String.valueOf(port), filePath};
		}
		catch(MalformedURLException e) {
			throw new IllegalArgumentException(
				"federated address `" + input + "` does not fit required URL pattern of \"host:port/directory\"", e);
		}
	}

	public static void federateMatrix(CacheableData<?> output, List<Pair<FederatedRange, FederatedData>> workers) {
		federateMatrix(output, workers, null);
	}

	public static void federateMatrix(CacheableData<?> output, List<Pair<FederatedRange, FederatedData>> workers, CacheBlock<?>[] blocks) {

		List<Pair<FederatedRange, FederatedData>> fedMapping = new ArrayList<>();
		for(Pair<FederatedRange, FederatedData> e : workers)
			fedMapping.add(e);
		List<Pair<FederatedData, Future<FederatedResponse>>> idResponses = new ArrayList<>();
		long id = FederationUtils.getNextFedDataID();
		boolean rowPartitioned = true;
		boolean colPartitioned = true;
		int k = 0;
		for(Pair<FederatedRange, FederatedData> entry : fedMapping) {
			FederatedRange range = entry.getKey();
			FederatedData value = entry.getValue();
			if(!value.isInitialized()) {
				long[] beginDims = range.getBeginDims();
				long[] endDims = range.getEndDims();
				long[] dims = output.getDataCharacteristics().getDims();
				for(int i = 0; i < dims.length; i++)
					dims[i] = endDims[i] - beginDims[i];
				if(blocks == null || blocks.length == 0)
					idResponses.add(new ImmutablePair<>(value, value.initFederatedData(id)));
				else
					idResponses.add(new ImmutablePair<>(value, value.initFederatedDataFromLocal(id, blocks[k++])));
			}
			rowPartitioned &= (range.getSize(1) == output.getNumColumns());
			colPartitioned &= (range.getSize(0) == output.getNumRows());
		}
		try {
			int timeout = ConfigurationManager.getDMLConfig()
				.getIntValue(DMLConfig.DEFAULT_FEDERATED_INITIALIZATION_TIMEOUT);
			if( LOG.isDebugEnabled() )
				LOG.debug("Federated Initialization with timeout: " + timeout);
			for(Pair<FederatedData, Future<FederatedResponse>> idResponse : idResponses) {
				// wait for initialization and check dimensions
				FederatedResponse re = idResponse.getRight().get(timeout, TimeUnit.SECONDS);
				DataCharacteristics dc = (DataCharacteristics) re.getData()[1];
				if( dc.getRows() > output.getNumRows() || dc.getCols() > output.getNumColumns() )
					throw new DMLRuntimeException("Invalid federated meta data: "
						+ output.getDataCharacteristics()+" vs federated response: "+dc);
			}
		}
		catch(TimeoutException e) {
			throw new DMLRuntimeException("Federated Initialization timeout exceeded", e);
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Federation initialization failed", e);
		}
		output.getDataCharacteristics().setNonZeros(-1);
		output.getDataCharacteristics().setBlocksize(ConfigurationManager.getBlocksize());
		output.setFedMapping(new FederationMap(id, fedMapping));

		output.getFedMapping().setType(rowPartitioned &&
			colPartitioned ? FType.FULL : rowPartitioned ? FType.ROW : colPartitioned ? FType.COL : FType.OTHER);

		if(LOG.isDebugEnabled())
			LOG.debug("Fed map Inited:" + output.getFedMapping());
	}

	public static void federateFrame(FrameObject output, List<Pair<FederatedRange, FederatedData>> workers) {
		federateFrame(output, workers, null);
	}

	public static void federateFrame(FrameObject output, List<Pair<FederatedRange, FederatedData>> workers, CacheBlock<?>[] blocks) {
		List<Pair<FederatedRange, FederatedData>> fedMapping = new ArrayList<>();
		for(Pair<FederatedRange, FederatedData> e : workers)
			fedMapping.add(e);
		// we want to wait for the futures with the response containing varIDs and the schemas of the frames
		// on the distributed workers. We need the FederatedData, the starting column of the sub frame (for the schema)
		// and the future for the response
		List<Pair<FederatedData, Pair<Integer, Future<FederatedResponse>>>> idResponses = new ArrayList<>();
		long id = FederationUtils.getNextFedDataID();
		boolean rowPartitioned = true;
		boolean colPartitioned = true;
		int k = 0;
		for(Pair<FederatedRange, FederatedData> entry : fedMapping) {
			FederatedRange range = entry.getKey();
			FederatedData value = entry.getValue();
			if(!value.isInitialized()) {
				long[] beginDims = range.getBeginDims();
				long[] endDims = range.getEndDims();
				long[] dims = output.getDataCharacteristics().getDims();
				for(int i = 0; i < dims.length; i++) {
					dims[i] = endDims[i] - beginDims[i];
				}
				if(blocks == null || blocks.length == 0)
					idResponses.add(
						new ImmutablePair<>(value, new ImmutablePair<>((int) beginDims[1], value.initFederatedData(id))));
				else
					idResponses.add(
						new ImmutablePair<>(value, new ImmutablePair<>((int) beginDims[1], value.initFederatedDataFromLocal(id, blocks[k++]))));
			}
			rowPartitioned &= (range.getSize(1) == output.getNumColumns());
			colPartitioned &= (range.getSize(0) == output.getNumRows());
		}
		// columns are definitely in int range, because we throw an DMLRuntime Exception in `processInstruction` else
		Types.ValueType[] schema = new Types.ValueType[(int) output.getNumColumns()];
		Arrays.fill(schema, Types.ValueType.UNKNOWN);
		try {
			for(Pair<FederatedData, Pair<Integer, Future<FederatedResponse>>> idResponse : idResponses) {
				FederatedData fedData = idResponse.getLeft();
				FederatedResponse response = idResponse.getRight().getRight().get();
				int startCol = idResponse.getRight().getLeft();
				handleFedFrameResponse(schema, fedData, response, startCol);
				DataCharacteristics dc = (DataCharacteristics) response.getData()[2];
				if( dc.getRows() > output.getNumRows() || dc.getCols() > output.getNumColumns() )
					throw new DMLRuntimeException("Invalid federated meta data: "
						+ output.getDataCharacteristics()+" vs federated response: "+dc);
			}
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Federation initialization failed", e);
		}
		output.getDataCharacteristics().setNonZeros(output.getNumColumns() * output.getNumRows());
		output.setSchema(schema);
		output.setFedMapping(new FederationMap(id, fedMapping));
		output.getFedMapping().setType(rowPartitioned &&
			colPartitioned ? FType.FULL : rowPartitioned ? FType.ROW : colPartitioned ? FType.COL : FType.OTHER);

		if(LOG.isDebugEnabled())
			LOG.debug("Fed map Inited: " + output.getFedMapping());
	}

	private static void handleFedFrameResponse(Types.ValueType[] schema, FederatedData federatedData,
		FederatedResponse response, int startColumn) {
		try {
			// Index 0 is the varID, Index 1 is the schema of the frame
			Object[] data = response.getData();
			federatedData.setVarID((Long) data[0]);
			// copy the schema
			Types.ValueType[] range_schema = (Types.ValueType[]) data[1];
			for(int i = 0; i < range_schema.length; i++) {
				Types.ValueType vType = range_schema[i];
				int schema_index = startColumn + i;
				if(schema[schema_index] != Types.ValueType.UNKNOWN && schema[schema_index] != vType)
					throw new DMLRuntimeException("federated Frame schemas mismatch");
				else
					schema[schema_index] = vType;
			}
		}
		catch(Exception e) {
			throw new DMLRuntimeException("Exception in frame response from federated worker.", e);
		}
	}

	@Override
	public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
		String type = ec.getScalarInput(_type).getStringValue();
		ListObject addresses = ec.getListObject(_addresses.getName());
		ListObject ranges = ec.getListObject(_ranges.getName());
		LineageItem[] liInputs = new LineageItem[addresses.getLength()];

		for(int i = 0; i < addresses.getLength(); i++) {
			Data addressData = addresses.getData().get(i);
			if(addressData instanceof StringObject) {
				String address = ((StringObject)addressData).getStringValue();
				// get beginning and end of data ranges
				List<Data> rangesData = ranges.getData();
				List<Data> beginDimsData = ((ListObject) rangesData.get(i*2)).getData();
				List<Data> endDimsData = ((ListObject) rangesData.get(i*2+1)).getData();
				String rl = ((ScalarObject)beginDimsData.get(0)).getStringValue();
				String cl = ((ScalarObject)beginDimsData.get(1)).getStringValue();
				String ru = ((ScalarObject)endDimsData.get(0)).getStringValue();
				String cu = ((ScalarObject)endDimsData.get(1)).getStringValue();
				// form a string with all the information and create a lineage item
				String data = InstructionUtils.concatOperands(type, address, rl, cl, ru, cu);
				liInputs[i] = new LineageItem(data);
			}
			else {
				throw new DMLRuntimeException("federated instruction only takes strings as addresses");
			}
		}
		return Pair.of(_output.getName(), new LineageItem(getOpcode(), liInputs));
	}
}
