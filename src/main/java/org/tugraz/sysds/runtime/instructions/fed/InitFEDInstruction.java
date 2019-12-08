/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.instructions.fed;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedData;
import org.tugraz.sysds.runtime.controlprogram.federated.FederatedRange;
import org.tugraz.sysds.runtime.instructions.InstructionUtils;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.ListObject;
import org.tugraz.sysds.runtime.instructions.cp.ScalarObject;
import org.tugraz.sysds.runtime.instructions.cp.StringObject;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class InitFEDInstruction extends FEDInstruction {
	private CPOperand _addresses, _ranges, _output;
	
	public InitFEDInstruction(CPOperand addresses, CPOperand ranges, CPOperand out, String opcode, String instr) {
		super(FEDType.Init, opcode, instr);
		_addresses = addresses;
		_ranges = ranges;
		_output = out;
	}
	
	public static InitFEDInstruction parseInstruction(String str) {
		String[] parts = InstructionUtils.getInstructionPartsWithValueType(str);
		// We need 3 parts: Opcode, Addresses (list of Strings with url/ip:port/filepath), ranges and the output Operand
		if( parts.length != 4 )
			throw new DMLRuntimeException("Invalid number of operands in federated instruction: " + str);
		String opcode = parts[0];
		
		CPOperand addresses, ranges, out;
		addresses = new CPOperand(parts[1]);
		ranges = new CPOperand(parts[2]);
		out = new CPOperand(parts[3]);
		return new InitFEDInstruction(addresses, ranges, out, opcode, str);
	}
	
	@Override
	public void processInstruction(ExecutionContext ec) {
		ListObject addresses = ec.getListObject(_addresses.getName());
		ListObject ranges = ec.getListObject(_ranges.getName());
		List<Pair<FederatedRange, FederatedData>> feds = new ArrayList<>();
		
		if( addresses.getLength() * 2 != ranges.getLength() )
			throw new DMLRuntimeException("Federated read needs twice the amount of addresses as ranges " +
					"(begin and end): addresses=" + addresses.getLength() + " ranges=" + ranges.getLength());
		
		long[] usedDims = new long[]{0, 0};
		for (int i = 0; i < addresses.getLength(); i++) {
			Data addressData = addresses.getData().get(i);
			if( addressData instanceof StringObject ) {
				String address = ((StringObject) addressData).getStringValue();
				// We split address into url/ip, the port and filepath of file to read
				String urlRegex = "^([-a-zA-Z0-9@]+(?:\\.[a-zA-Z0-9]+)*)";
				String portRegex = ":([0-9]+)";
				String filepathRegex = "((?:/[\\w-]+)*/[\\w-.]+)/?$";
				Pattern compiled = Pattern.compile(urlRegex + portRegex + filepathRegex);
				Matcher matcher = compiled.matcher(address);
				if( matcher.matches() ) {
					// matches: 0 whole match, 1 host address, 2 port, 3 filepath
					String host = matcher.group(1);
					int port = Integer.parseInt(matcher.group(2));
					String filepath = matcher.group(3).substring(1);
					// get begin and end ranges
					List<Data> rangesData = ranges.getData();
					Data beginData = rangesData.get(i * 2);
					Data endData = rangesData.get(i * 2 + 1);
					if( beginData.getDataType() != Types.DataType.LIST || endData.getDataType() != Types.DataType.LIST )
						throw new DMLRuntimeException("Federated read ranges (lower, upper) have to be lists of dimensions");
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
						FederatedData federatedData = new FederatedData(
								new InetSocketAddress(InetAddress.getByName(host), port), filepath);
						feds.add(new ImmutablePair<>(new FederatedRange(beginDims, endDims), federatedData));
					}
					catch (UnknownHostException e) {
						throw new DMLRuntimeException("federated host was unknown: " + host);
					}
				}
				else {
					throw new DMLRuntimeException("federated address `" + address + "` does not fit required pattern " +
							"of \"host:port/directory\"");
				}
			}
			else {
				throw new DMLRuntimeException("federated instruction only takes strings as addresses");
			}
		}
		MatrixObject matrixObject = ec.getMatrixObject(_output);
		matrixObject.getDataCharacteristics().setRows(usedDims[0]).setCols(usedDims[1]);
		matrixObject.federate(feds);
	}
}
