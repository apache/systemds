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

package org.apache.sysds.runtime.lineage;


import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.parser.ParseException;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionParser;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LineageParser
{
	public final static LineageTokenizer lineageTraceTokenizer = new LineageTokenizer();
	
	static {
		lineageTraceTokenizer.add("\\(");
		lineageTraceTokenizer.add("[0-9]+", "id");
		lineageTraceTokenizer.add("\\) \\(");
		lineageTraceTokenizer.add("L|C|I", "type");
		lineageTraceTokenizer.add("\\) ");
		lineageTraceTokenizer.add(".+", "representation");
	}
	
	public static LineageItem parseLineageTrace(String str) {
		return parseLineageTrace(str, null);
	}
	
	public static LineageItem parseLineageTrace(String str, String name) {
		ExecutionContext ec = ExecutionContextFactory.createContext();
		LineageItem li = null;
		Map<Long, LineageItem> map = new HashMap<>();
		
		for (String line : str.split("\\r?\\n")) {
			li = null;
			Map<String, String> tokens = lineageTraceTokenizer.tokenize(line);
			Long id = Long.valueOf(tokens.get("id"));
			LineageItem.LineageItemType type = LineageItemUtils.getType(tokens.get("type"));
			String representation = tokens.get("representation");
			
			switch (type) {
				case Creation:
					Instruction inst = InstructionParser.parseSingleInstruction(representation);
					if (!(inst instanceof LineageTraceable))
						throw new ParseException("Invalid Instruction (" + inst.getOpcode() + ") traced");
					Pair<String,LineageItem> item = ((LineageTraceable) inst).getLineageItem(ec);
					if (item == null)
						throw new ParseException("Instruction without output (" + inst.getOpcode() + ") not supported");
					
					li = new LineageItem(id, item.getValue());
					break;
				
				case Literal:
					li = new LineageItem(id, representation);
					break;
				
				case Instruction:
					li = parseLineageInstruction(id, representation, map, name);
					break;
				
				default:
					throw new ParseException("Invalid LineageItemType given");
			}
			map.put(id, li);
		}
		return li;
	}
	
	private static LineageItem parseLineageInstruction(Long id, String str, Map<Long, LineageItem> map, String name) {
		String[] tokens = str.split(" ");
		if (tokens.length < 2)
			throw new ParseException("Invalid length ot lineage item "+tokens.length+".");
		String opcode = tokens[0];

		if (opcode.startsWith(LineageItemUtils.LPLACEHOLDER)) {
			// Convert this to a leaf node (creation type)
			String data = opcode;
			return new LineageItem(id, data, "Create"+opcode);
		}

		ArrayList<LineageItem> inputs = new ArrayList<>();
		for( int i=1; i<tokens.length; i++ ) {
			String token = tokens[i];
			if (token.startsWith("(") && token.endsWith(")")) {
				token = token.substring(1, token.length()-1); //rm parentheses
				inputs.add(map.get(Long.valueOf(token)));
			} else
				throw new ParseException("Invalid format for LineageItem reference");
		}
		return new LineageItem(id, "", opcode, inputs.toArray(new LineageItem[0]));
	}
	
	public static LineageItem parseLineageTraceDedup(String str) {
		LineageItem li = null;
		Map<Long, Map<String, LineageItem>> patchLiMap = new HashMap<>();
		str.replaceAll("\r\n", "\n");
		String[] allPatches = str.split("\n\n");
		for (String patch : allPatches) {
			String[] headBody = patch.split("\r\n|\r|\n", 2);
			// Parse the header
			String[] parts = headBody[0].split(LineageItemUtils.DEDUP_DELIM);
			// e.g. patch_R_SB15_1
			// Deserialize the patch
			LineageItem patchLi = parseLineageTrace(headBody[1]);
			Long pathId = Long.parseLong(parts[3]);
			// Map the pathID and the DAG root name to the deserialized DAG.
			if (!patchLiMap.containsKey(pathId)) {
				patchLiMap.put(pathId, new HashMap<>());
			}
			patchLiMap.get(pathId).put(parts[1], patchLi);
			// TODO: handle multiple loops
		}
		return li;
	}
}