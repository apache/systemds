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

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRange;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.federated.FederationMap.FType;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Future;

public class InitFEDInstruction extends FEDInstruction {
	
	public static final String FED_MATRIX_IDENTIFIER = "matrix";
	public static final String FED_FRAME_IDENTIFIER = "frame";

	private CPOperand _type, _addresses, _ranges, _output;

	public InitFEDInstruction(CPOperand type, CPOperand addresses, CPOperand ranges, CPOperand out, String opcode, String instr) {
		super(FEDType.Init, opcode, instr);
		_type = type;
		_addresses = addresses;
		_ranges = ranges;
		_output = out;
	}

	public static InitFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// We need 5 parts: Opcode, Type (Frame/Matrix), Addresses (list of Strings with
		// url/ip:port/filepath), ranges and the output Operand
		if (parts.length != 5)
			throw new DMLRuntimeException("Invalid number of operands in federated instruction: " + str);
		String opcode = parts[0];

		CPOperand type, addresses, ranges, out;
		type = new CPOperand(parts[1]);
		addresses = new CPOperand(parts[2]);
		ranges = new CPOperand(parts[3]);
		out = new CPOperand(parts[4]);
		return new InitFEDInstruction(type, addresses, ranges, out, opcode, str);
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		String type = ec.getScalarInput(_type).getStringValue();
		ListObject addresses = ec.getListObject(_addresses.getName());
		ListObject ranges = ec.getListObject(_ranges.getName());
		List<Pair<FederatedRange, FederatedData>> feds = new ArrayList<>();

		if (addresses.getLength() * 2 != ranges.getLength())
			throw new DMLRuntimeException("Federated read needs twice the amount of addresses as ranges "
				+ "(begin and end): addresses=" + addresses.getLength() + " ranges=" + ranges.getLength());
		
		Types.DataType fedDataType;
		if (type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER))
			fedDataType = Types.DataType.MATRIX;
		else if (type.equalsIgnoreCase(FED_FRAME_IDENTIFIER))
			fedDataType = Types.DataType.FRAME;
		else
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");
		
		long[] usedDims = new long[] { 0, 0 };
		for (int i = 0; i < addresses.getLength(); i++) {
			Data addressData = addresses.getData().get(i);
			if (addressData instanceof StringObject) {
				// We split address into url/ip, the port and file path of file to read
				String[] parsedValues = parseURL(((StringObject) addressData).getStringValue());
				String host = parsedValues[0];
				int port = Integer.parseInt(parsedValues[1]);
				String filePath = parsedValues[2];
				// get beginning and end of data ranges
				List<Data> rangesData = ranges.getData();
				Data beginData = rangesData.get(i * 2);
				Data endData = rangesData.get(i * 2 + 1);
				if (beginData.getDataType() != Types.DataType.LIST || endData.getDataType() != Types.DataType.LIST)
					throw new DMLRuntimeException(
						"Federated read ranges (lower, upper) have to be lists of dimensions");
				List<Data> beginDimsData = ((ListObject) beginData).getData();
				List<Data> endDimsData = ((ListObject) endData).getData();

				// fill begin and end dims
				long[] beginDims = new long[beginDimsData.size()];
				long[] endDims = new long[beginDims.length];
				for (int d = 0; d < beginDims.length; d++) {
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
				catch (UnknownHostException e) {
					throw new DMLRuntimeException("federated host was unknown: " + host);
				}
			}
			else {
				throw new DMLRuntimeException("federated instruction only takes strings as addresses");
			}
		}
		if (type.equalsIgnoreCase(FED_MATRIX_IDENTIFIER)) {
			MatrixObject output = ec.getMatrixObject(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateMatrix(output, feds);
		}
		else if (type.equalsIgnoreCase(FED_FRAME_IDENTIFIER)) {
			if (usedDims[1] > Integer.MAX_VALUE)
				throw new DMLRuntimeException("federated Frame can not have more than max int columns, because the " +
						"schema can only be max int length");
			FrameObject output = ec.getFrameObject(_output);
			output.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
			federateFrame(output, feds);
		}
		else {
			throw new DMLRuntimeException("type \"" + type + "\" non valid federated type");
		}
	}

	public static String[] parseURL(String input) {
		try {
			// Artificially making it http protocol. 
			// This is to avoid malformed address error in the URL passing.
			// TODO: Construct new protocol name for Federated communication
			URL address = new URL("http://" + input);
			String host = address.getHost();
			if (host.length() == 0)
				throw new IllegalArgumentException("Missing Host name for federated address");
			// The current system does not support ipv6, only ipv4.
			// TODO: Support IPV6 address for Federated communication
			String ipRegex = "^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$";
			if (host.matches("^\\d+\\.\\d+\\.\\d+\\.\\d+$") && !host.matches(ipRegex))
				throw new IllegalArgumentException("Input Host address looks like an IP address but is outside range");
			int port = address.getPort();
			if (port == -1)
				port = DMLConfig.DEFAULT_FEDERATED_PORT;
			String filePath = address.getPath();
			if (filePath.length() <= 1)
				throw new IllegalArgumentException("Missing File path for federated address");
			// Remove the first character making the path Dynamic from the location of the worker.
			// This is in contrast to before where it was static paths
			filePath = filePath.substring(1);
			// To make static file paths use double "//" EG:
			// example.dom//staticFile.txt
			// example.dom/dynamicFile.txt
			if (address.getQuery() != null)
				throw new IllegalArgumentException("Query is not supported");

			if (address.getRef() != null)
				throw new IllegalArgumentException("Reference is not supported");
			
			return new String[] { host, String.valueOf(port), filePath };
		}
		catch (MalformedURLException e) {
			throw new IllegalArgumentException("federated address `" + input
				+ "` does not fit required URL pattern of \"host:port/directory\"", e);
		}
	}

	public void federateMatrix(MatrixObject output, List<Pair<FederatedRange, FederatedData>> workers) {
		Map<FederatedRange, FederatedData> fedMapping = new TreeMap<>();
		for (Pair<FederatedRange, FederatedData> t : workers) {
			fedMapping.put(t.getLeft(), t.getRight());
		}
		List<Pair<FederatedData, Future<FederatedResponse>>> idResponses = new ArrayList<>();
		long id = FederationUtils.getNextFedDataID();
		boolean rowPartitioned = true;
		boolean colPartitioned = true;
		for (Map.Entry<FederatedRange, FederatedData> entry : fedMapping.entrySet()) {
			FederatedRange range = entry.getKey();
			FederatedData value = entry.getValue();
			if (!value.isInitialized()) {
				long[] beginDims = range.getBeginDims();
				long[] endDims = range.getEndDims();
				long[] dims = output.getDataCharacteristics().getDims();
				for (int i = 0; i < dims.length; i++)
					dims[i] = endDims[i] - beginDims[i];
				idResponses.add(new ImmutablePair<>(value, value.initFederatedData(id)));
			}
			rowPartitioned &= (range.getSize(1) == output.getNumColumns());
			colPartitioned &= (range.getSize(0) == output.getNumRows()); 
		}
		try {
			for (Pair<FederatedData, Future<FederatedResponse>> idResponse : idResponses)
				idResponse.getRight().get(); //wait for initialization
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Federation initialization failed", e);
		}
		output.getDataCharacteristics().setNonZeros(-1);
		output.getDataCharacteristics().setBlocksize(ConfigurationManager.getBlocksize());
		output.setFedMapping(new FederationMap(id, fedMapping));
		output.getFedMapping().setType(rowPartitioned ? FType.ROW : colPartitioned ? FType.COL : FType.OTHER);
	}
	
	public void federateFrame(FrameObject output, List<Pair<FederatedRange, FederatedData>> workers) {
		Map<FederatedRange, FederatedData> fedMapping = new TreeMap<>();
		for (Pair<FederatedRange, FederatedData> t : workers) {
			fedMapping.put(t.getLeft(), t.getRight());
		}
		// we want to wait for the futures with the response containing varIDs and the schemas of the frames
		// on the distributed workers. We need the FederatedData, the starting column of the sub frame (for the schema)
		// and the future for the response
		List<Pair<FederatedData, Pair<Integer, Future<FederatedResponse>>>> idResponses = new ArrayList<>();
		long id = FederationUtils.getNextFedDataID();
		for (Map.Entry<FederatedRange, FederatedData> entry : fedMapping.entrySet()) {
			FederatedRange range = entry.getKey();
			FederatedData value = entry.getValue();
			if (!value.isInitialized()) {
				long[] beginDims = range.getBeginDims();
				long[] endDims = range.getEndDims();
				long[] dims = output.getDataCharacteristics().getDims();
				for (int i = 0; i < dims.length; i++) {
					dims[i] = endDims[i] - beginDims[i];
				}
				idResponses.add(new ImmutablePair<>(value, new ImmutablePair<>((int) beginDims[1], value.initFederatedData(id))));
			}
		}
		// columns are definitely in int range, because we throw an DMLRuntime Exception in `processInstruction` else
		Types.ValueType[] schema = new Types.ValueType[(int) output.getNumColumns()];
		Arrays.fill(schema, Types.ValueType.UNKNOWN);
		try {
			for (Pair<FederatedData, Pair<Integer, Future<FederatedResponse>>> idResponse : idResponses) {
				FederatedData fedData = idResponse.getLeft();
				FederatedResponse response = idResponse.getRight().getRight().get();
				int startCol = idResponse.getRight().getLeft();
				handleFedFrameResponse(schema, fedData, response, startCol);
			}
		}
		catch (Exception e) {
			throw new DMLRuntimeException("Federation initialization failed", e);
		}
		output.getDataCharacteristics().setNonZeros(output.getNumColumns() * output.getNumRows());
		output.setSchema(schema);
		output.setFedMapping(new FederationMap(id, fedMapping));
	}
	
	private static void handleFedFrameResponse(Types.ValueType[] schema, FederatedData federatedData,
		FederatedResponse response, int startColumn) {
		try {
			// Index 0 is the varID, Index 1 is the schema of the frame
			Object[] data = response.getData();
			federatedData.setVarID((Long) data[0]);
			// copy the
			Types.ValueType[] range_schema = (Types.ValueType[]) data[1];
			for(int i = 0; i < range_schema.length; i++) {
				Types.ValueType vType = range_schema[i];
				int schema_index = startColumn + i;
				if(schema[schema_index] != Types.ValueType.UNKNOWN && schema[schema_index] != vType)
					throw new DMLRuntimeException("federated Frame schemas mismatch");
				else
					schema[schema_index] = vType;
			}
		} catch (Exception e){
			throw new DMLRuntimeException("Exception in frame response from federated worker.", e);
		}
	}
}
