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
 */


package org.tugraz.sysds.runtime.lineage;


import org.tugraz.sysds.parser.ParseException;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.InstructionParser;
import org.tugraz.sysds.runtime.instructions.cp.CPOperand;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class LineageParser {
	
	public static LineageTokenizer lineageTraceTokenizer = new LineageTokenizer();
	
	static {
		lineageTraceTokenizer.add("\\(");
		lineageTraceTokenizer.add("[0-9]+", "id");
		lineageTraceTokenizer.add("\\) \\(");
		lineageTraceTokenizer.add("L|C|I", "type");
		lineageTraceTokenizer.add("\\) ");
		lineageTraceTokenizer.add(".+", "representation");
	}
	
	public static LineageItem parseLineageTrace(String str) {
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
					
					li = new LineageItem(id, ((LineageTraceable) inst).getLineageItem());
					break;
				
				case Literal:
					li = new LineageItem(id, new CPOperand(representation).getName(), representation);
					break;
				
				case Instruction:
					li = parseLineageInstruction(id, representation, map);
					break;
				
				default:
					throw new ParseException("Invalid LineageItemType given");
			}
			map.put(id, li);
		}
		return li;
	}
	
	private static LineageItem parseLineageInstruction(Long id, String str, Map<Long, LineageItem> map) {
		ArrayList<LineageItem> inputs = new ArrayList<>();
		String[] tokens = str.split(" ");
		if (tokens.length < 2)
			throw new ParseException("Invalid length ot lineage item "+tokens.length+".");
		String opcode = tokens[0];
		for( int i=1; i<tokens.length; i++ ) {
			String token = tokens[i];
			if (token.startsWith("(") && token.endsWith(")")) {
				token = token.substring(1, token.length()-1); //rm parentheses
				inputs.add(map.get(Long.valueOf(token)));
			} else
				throw new ParseException("Invalid format for LineageItem reference");
		}
		return new LineageItem(id, null, inputs, opcode);
	}
}