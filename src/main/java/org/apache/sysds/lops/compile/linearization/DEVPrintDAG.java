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

package org.apache.sysds.lops.compile.linearization;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

import org.apache.sysds.lops.Lop;

//DEBUGGING PURPOSES WHILE DEVELOPING
public class DEVPrintDAG {
    
	static void asGraphviz(String filePrefix, List<Lop> v) {
	
		StringBuilder sb = new StringBuilder();
		sb.append("digraph G {\n");
	
		for (int i = 0; i < v.size(); i++) {
			Lop l = v.get(i);

			String cutted_l = l.toString().substring(0, Math.min(l.toString().length(), 75));
			String toString[] = cutted_l.replaceAll("((.*?\\s){" + (2) + "})", "$1@").split("@");
			String toString2 = String.join("\n", toString);
			sb.append(l.getID() +  " [label=\"" + l.getID() + "::" + i + "::" + l.getPipelineID() + " " + l.getType() + " " + l.getLevel() +  ":\n " + toString2 +"\", style=filled, color="+ DEVPrintDAG.getColorName(l.getPipelineID()) +"];\n");

			for (Lop in : l.getInputs()) {
				sb.append(in.getID() + " -> " + l.getID() + ";\n");
				//sb.append(l.getID() + " -> " + in.getID() + ";\n");
			}
		}

		sb.append("}\n");

		String currentProcessId = String.valueOf(ProcessHandle.current().pid());

		String folderName = "graphs/graphs_" + currentProcessId;
		File folder = new File(folderName);

		if (!folder.exists()) {
			folder.mkdirs();
		}

		String filename = folderName + "/" + filePrefix + "_" + v.size() + "_" + System.currentTimeMillis() + ".dot";
		try {
			Files.write(Paths.get(filename), sb.toString().getBytes());
		} catch (Exception e) {
			e.printStackTrace();
		}

	}

    //Colors only correctly work with VSCode extension: Graphviz Interactive Preview
	private static String getColorName(int value) {
		String[] colors = {
			"red", "orange", "yellow", "green", "blue", "indigo", "violet",
			"teal", "purple", "pink", "brown", "beige", "cyan", "turquoise",
			"lime", "olive", "mediumseagreen", "azure", "aquamarine", "magenta",
			"silver", "gold", "goldenrod4", "sienna2", "deeppink3", "darkolivegreen", "grey52",
			"darkslateblue", "darkkhaki", "cornflowerblue", "yellowgreen", "chartreuse2", "navyblue"
		};
	
		if (value < 0) {
			return "lightgrey";
		}
	
		return colors[value % 32];
	}

}
