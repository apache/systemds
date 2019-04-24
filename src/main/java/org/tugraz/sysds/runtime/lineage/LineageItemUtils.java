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

import org.tugraz.sysds.runtime.lineage.LineageItem.LineageItemType;
import org.tugraz.sysds.runtime.DMLRuntimeException;

import java.util.stream.Collectors;

public class LineageItemUtils {
	
	public static LineageItemType getType(LineageItem li) {
		if (li.isLeaf() && li.isInstruction())
			return LineageItemType.Creation;
		else if (li.isLeaf() && !li.isInstruction())
			return LineageItemType.Literal;
		else if (!li.isLeaf() && li.isInstruction())
			return LineageItemType.Instruction;
		else
			throw new DMLRuntimeException("An inner node could not be a literal!");
	}
	
	public static LineageItemType getType(String str) {
		if (str.length() == 1) {
			switch (str) {
				case "C":
					return LineageItemType.Creation;
				case "L":
					return LineageItemType.Literal;
				case "I":
					return LineageItemType.Instruction;
				default:
					throw new DMLRuntimeException("Unknown LineageItemType given!");
			}
		} else
			throw new DMLRuntimeException("Unknown LineageItemType given!");
	}
	
	private static String getString(LineageItemType lit) {
		switch (lit) {
			case Creation:
				return "C";
			case Literal:
				return "L";
			case Instruction:
				return "I";
			default:
				throw new DMLRuntimeException("Unknown LineageItemType given!");
		}
	}
	
	private static String getString(LineageItem li) {
		return getString(getType(li));
	}
	
	public static String explainSingleLineageItem(LineageItem li) {
		StringBuilder sb = new StringBuilder();
		sb.append("(").append(li.getId()).append(") ");
		sb.append("(").append(getString(li)).append(") ");
		
		if (li.isLeaf()) {
			sb.append(li.getRepresentation()).append(" ");
		} else {
			sb.append(li.getOpcode()).append(" ");
			String ids = li.getInputs().stream()
					.map(i -> String.format("(%d)", i.getId()))
					.collect(Collectors.joining(" "));
			sb.append(ids);
		}
		return sb.toString().trim();
	}
}
