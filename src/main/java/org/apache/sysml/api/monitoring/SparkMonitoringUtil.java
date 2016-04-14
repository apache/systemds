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
package org.apache.sysml.api.monitoring;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import scala.collection.Seq;
import scala.xml.Node;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.spark.SPInstruction;
import org.apache.sysml.runtime.instructions.spark.functions.SparkListener;

/**
 * Usage guide:
 * MLContext mlCtx = new MLContext(sc, true);
 * mlCtx.register...
 * mlCtx.execute(...)
 * mlCtx.getMonitoringUtil().getRuntimeInfoInHTML("runtime.html");
 */
public class SparkMonitoringUtil {
	// ----------------------------------------------------
	// For VLDB Demo:
	private Multimap<Location, String> instructions = TreeMultimap.create();
	private Multimap<String, Integer> stageIDs = TreeMultimap.create();  // instruction -> stageIds
	private Multimap<String, Integer> jobIDs = TreeMultimap.create();  // instruction -> jobIds
	private HashMap<String, String> lineageInfo = new HashMap<String, String>();	// instruction -> lineageInfo
	private HashMap<String, Long> instructionCreationTime = new HashMap<String, Long>();
	
	private Multimap<Integer, String> rddInstructionMapping = TreeMultimap.create();
	
	private HashSet<String> getRelatedInstructions(int stageID) {
		HashSet<String> retVal = new HashSet<String>();
		if(_sparkListener != null) {
			ArrayList<Integer> rdds = _sparkListener.stageRDDMapping.get(stageID);
			for(Integer rddID : rdds) {
				retVal.addAll(rddInstructionMapping.get(rddID));
			}
		}
		return retVal;
	}
	
	private SparkListener _sparkListener = null;
	public SparkListener getSparkListener() {
		return _sparkListener;
	}
	
	private String explainOutput = "";
	
	public String getExplainOutput() {
		return explainOutput;
	}

	public void setExplainOutput(String explainOutput) {
		this.explainOutput = explainOutput;
	}

	public SparkMonitoringUtil(SparkListener sparkListener) {
		_sparkListener = sparkListener;
	}
	
	public void addCurrentInstruction(SPInstruction inst) {
		if(_sparkListener != null) {
			_sparkListener.addCurrentInstruction(inst);
		}
	}
	
	public void addRDDForInstruction(SPInstruction inst, Integer rddID) {
		this.rddInstructionMapping.put(rddID, getInstructionString(inst));
	}
	
	public void removeCurrentInstruction(SPInstruction inst) {
		if(_sparkListener != null) {
			_sparkListener.removeCurrentInstruction(inst);
		}
	}
	
	public void setDMLString(String dmlStr) {
		this.dmlStrForMonitoring = dmlStr;
	}
	
	public void resetMonitoringData() {
		if(_sparkListener != null && _sparkListener.stageDAGs != null)
			_sparkListener.stageDAGs.clear();
		if(_sparkListener != null && _sparkListener.stageTimeline != null)
			_sparkListener.stageTimeline.clear();
	}
	
	// public Multimap<Location, String> hops = ArrayListMultimap.create(); TODO:
	private String dmlStrForMonitoring = null;
	public void getRuntimeInfoInHTML(String htmlFilePath) throws DMLRuntimeException, IOException {
		String jsAndCSSFiles = "<script src=\"js/lodash.min.js\"></script>"
				+ "<script src=\"js/jquery-1.11.1.min.js\"></script>"
				+ "<script src=\"js/d3.min.js\"></script>"
				+ "<script src=\"js/bootstrap-tooltip.js\"></script>"
				+ "<script src=\"js/dagre-d3.min.js\"></script>"
				+ "<script src=\"js/graphlib-dot.min.js\"></script>"
				+ "<script src=\"js/spark-dag-viz.js\"></script>"
				+ "<script src=\"js/timeline-view.js\"></script>"
				+ "<script src=\"js/vis.min.js\"></script>"
				+ "<link rel=\"stylesheet\" href=\"css/bootstrap.min.css\">"
				+ "<link rel=\"stylesheet\" href=\"css/vis.min.css\">"
				+ "<link rel=\"stylesheet\" href=\"css/spark-dag-viz.css\">"
				+ "<link rel=\"stylesheet\" href=\"css/timeline-view.css\"> ";
		BufferedWriter bw = new BufferedWriter(new FileWriter(htmlFilePath));
		bw.write("<html><head>\n");
		bw.write(jsAndCSSFiles + "\n");
		bw.write("</head><body>\n<table border=1>\n");
		
		bw.write("<tr>\n");
		bw.write("<td><b>Position in script</b></td>\n");
		bw.write("<td><b>DML</b></td>\n");
		bw.write("<td><b>Instruction</b></td>\n");
		bw.write("<td><b>StageIDs</b></td>\n");
		bw.write("<td><b>RDD Lineage</b></td>\n");
		bw.write("</tr>\n");
		
		for(Location loc : instructions.keySet()) {
			String dml = getExpression(loc);
			
			// Sort the instruction with time - so as to separate recompiled instructions
			List<String> listInst = new ArrayList<String>(instructions.get(loc));
			Collections.sort(listInst, new InstructionComparator(instructionCreationTime));
			
			if(dml != null && dml.trim().length() > 1) {
				bw.write("<tr>\n");
				int rowSpan = listInst.size();
				bw.write("<td rowspan=\"" + rowSpan + "\">" + loc.toString() + "</td>\n");
				bw.write("<td rowspan=\"" + rowSpan + "\">" + dml + "</td>\n");
				boolean firstTime = true;
				for(String inst : listInst) {
					if(!firstTime)
						bw.write("<tr>\n");
					
					if(inst.startsWith("SPARK"))
						bw.write("<td style=\"color:red\">" + inst + "</td>\n");
					else if(isInterestingCP(inst))
						bw.write("<td style=\"color:blue\">" + inst + "</td>\n");
					else
						bw.write("<td>" + inst + "</td>\n");
					
					bw.write("<td>" + getStageIDAsString(inst) + "</td>\n");
					if(lineageInfo.containsKey(inst))
						bw.write("<td>" + lineageInfo.get(inst).replaceAll("\n", "<br />") + "</td>\n");
					else
						bw.write("<td></td>\n");
					
					bw.write("</tr>\n");
					firstTime = false;
				}
				
			}
			
		}
		
		bw.write("</table></body>\n</html>");
		bw.close();
	}
	
	private String getInQuotes(String str) {
		return "\"" + str + "\"";
	}
	private String getEscapedJSON(String json) {
		if(json == null)
			return "";
		else {
			return json
					//.replaceAll("\\\\", "\\\\\\")
					.replaceAll("\\t", "\\\\t")
					.replaceAll("/", "\\\\/")
					.replaceAll("\"", "\\\\\"")
					.replaceAll("\\r?\\n", "\\\\n");
		}
	}
	
	private long maxExpressionExecutionTime = 0;
	HashMap<Integer, Long> stageExecutionTimes = new HashMap<Integer, Long>();
	HashMap<String, Long> expressionExecutionTimes = new HashMap<String, Long>();
	HashMap<String, Long> instructionExecutionTimes = new HashMap<String, Long>();
	HashMap<Integer, HashSet<String>> relatedInstructionsPerStage = new HashMap<Integer, HashSet<String>>();
	private void fillExecutionTimes() {
		stageExecutionTimes.clear();
		expressionExecutionTimes.clear();
		for(Location loc : instructions.keySet()) {
			List<String> listInst = new ArrayList<String>(instructions.get(loc));
			long expressionExecutionTime = 0;
			
			for(String inst : listInst) {
				long instructionExecutionTime = 0;
				for(Integer stageId : stageIDs.get(inst)) {
					try {
						if(getStageExecutionTime(stageId) != null) {
							long stageExecTime = getStageExecutionTime(stageId);
							instructionExecutionTime += stageExecTime;
							expressionExecutionTime += stageExecTime;
							stageExecutionTimes.put(stageId, stageExecTime);
						}
					}
					catch(Exception e) {}

					relatedInstructionsPerStage.put(stageId, getRelatedInstructions(stageId));
				}
				instructionExecutionTimes.put(inst, instructionExecutionTime);
			}
			expressionExecutionTime /= listInst.size(); // average
			maxExpressionExecutionTime = Math.max(maxExpressionExecutionTime, expressionExecutionTime);
			expressionExecutionTimes.put(loc.toString(), expressionExecutionTime);
		}
		
		// Now fill empty instructions
		for(Entry<String, Long> kv : instructionExecutionTimes.entrySet()) {
			if(kv.getValue() == 0) {
				// Find all stages that contain this as related instruction
				long sumExecutionTime = 0;
				for(Entry<Integer, HashSet<String>> kv1 : relatedInstructionsPerStage.entrySet()) {
					if(kv1.getValue().contains(kv.getKey())) {
						sumExecutionTime += stageExecutionTimes.get(kv1.getKey());
					}
				}
				kv.setValue(sumExecutionTime);
			}
		}
		
		for(Location loc : instructions.keySet()) {
			if(expressionExecutionTimes.get(loc.toString()) == 0) {
				List<String> listInst = new ArrayList<String>(instructions.get(loc));
				long expressionExecutionTime = 0;
				for(String inst : listInst) {
					expressionExecutionTime += instructionExecutionTimes.get(inst);
				}
				expressionExecutionTime /= listInst.size(); // average
				maxExpressionExecutionTime = Math.max(maxExpressionExecutionTime, expressionExecutionTime);
				expressionExecutionTimes.put(loc.toString(), expressionExecutionTime);
			}
		}
		
	}
	
	/**
	 * Useful to avoid passing large String through Py4J
	 * @param fileName
	 * @throws DMLRuntimeException
	 * @throws IOException
	 */
	public void saveRuntimeInfoInJSONFormat(String fileName) throws DMLRuntimeException, IOException {
		String json = getRuntimeInfoInJSONFormat();
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		bw.write(json);
		bw.close();
	}
	
	public String getRuntimeInfoInJSONFormat() throws DMLRuntimeException, IOException {
		StringBuilder retVal = new StringBuilder("{\n");
		
		retVal.append(getInQuotes("dml") + ":" + getInQuotes(getEscapedJSON(dmlStrForMonitoring)) + ",\n");
		retVal.append(getInQuotes("expressions") + ":" + "[\n");
		
		boolean isFirstExpression = true;
		fillExecutionTimes();
		
		for(Location loc : instructions.keySet()) {
			String dml = getEscapedJSON(getExpressionInJSON(loc));
			
			if(dml != null) {
				// Sort the instruction with time - so as to separate recompiled instructions
				List<String> listInst = new ArrayList<String>(instructions.get(loc));
				Collections.sort(listInst, new InstructionComparator(instructionCreationTime));
				
				if(!isFirstExpression) {
					retVal.append(",\n");
				}
				retVal.append("{\n");
				isFirstExpression = false;
				
				retVal.append(getInQuotes("beginLine") + ":" + loc.beginLine + ",\n");
				retVal.append(getInQuotes("beginCol") + ":" + loc.beginCol + ",\n");
				retVal.append(getInQuotes("endLine") + ":" + loc.endLine + ",\n");
				retVal.append(getInQuotes("endCol") + ":" + loc.endCol + ",\n");
				
				long expressionExecutionTime = expressionExecutionTimes.get(loc.toString());
				retVal.append(getInQuotes("expressionExecutionTime") + ":" + expressionExecutionTime + ",\n");
				retVal.append(getInQuotes("expressionHeavyHitterFactor") + ":" + ((double)expressionExecutionTime / (double)maxExpressionExecutionTime) + ",\n");
				
				retVal.append(getInQuotes("expression") + ":" + getInQuotes(dml) + ",\n");
				
				retVal.append(getInQuotes("instructions") + ":" + "[\n");
			
				boolean firstTime = true;
				for(String inst : listInst) {
					
					if(!firstTime)
						retVal.append(", {");
					else
						retVal.append("{");
					
					if(inst.startsWith("SPARK")) {
						retVal.append(getInQuotes("isSpark") + ":" + "true,\n"); 
					}
					else if(isInterestingCP(inst)) {
						retVal.append(getInQuotes("isInteresting") + ":" + "true,\n");
					}
					
					retVal.append(getStageIDAsJSONString(inst) + "\n");
					if(lineageInfo.containsKey(inst)) {
						retVal.append(getInQuotes("lineageInfo") + ":" + getInQuotes(getEscapedJSON(lineageInfo.get(inst))) + ",\n");
					}
					
					retVal.append(getInQuotes("instruction") + ":" + getInQuotes(getEscapedJSON(inst)));
					retVal.append("}");
					firstTime = false;
				}
				
				retVal.append("]\n");
				retVal.append("}\n");
			}
			
		}
		
		return retVal.append("]\n}").toString();
	}
	
	
	private boolean isInterestingCP(String inst) {
		if(inst.startsWith("CP rmvar") || inst.startsWith("CP cpvar") || inst.startsWith("CP mvvar"))
			return false;
		else if(inst.startsWith("CP"))
			return true;
		else
			return false;
	}
	
	private String getStageIDAsString(String instruction) {
		String retVal = "";
		for(Integer stageId : stageIDs.get(instruction)) {
			String stageDAG = "";
			String stageTimeLine = "";
			
			if(getStageDAGs(stageId) != null) {
				stageDAG = getStageDAGs(stageId).toString();
			}
			
			if(getStageTimeLine(stageId) != null) {
				stageTimeLine = getStageTimeLine(stageId).toString();
			}
			
			retVal +=  "Stage:" + stageId + 
					" ("
					+ "<div>" 
						+ stageDAG.replaceAll("toggleDagViz\\(false\\)", "toggleDagViz(false, this)") 
					+ "</div>, "
					+ "<div id=\"timeline-" + stageId + "\">"
						+ stageTimeLine
							.replaceAll("drawTaskAssignmentTimeline\\(", "registerTimelineData(" + stageId + ", ")
							.replaceAll("class=\"expand-task-assignment-timeline\"",  "class=\"expand-task-assignment-timeline\" onclick=\"toggleStageTimeline(this)\"")
					+ "</div>"
					+ ")"; 
		}
		return retVal;
	}
	
	private String getStageIDAsJSONString(String instruction) {
		long instructionExecutionTime = instructionExecutionTimes.get(instruction);
		
		StringBuilder retVal = new StringBuilder(getInQuotes("instructionExecutionTime") + ":" + instructionExecutionTime + ",\n");
		
		boolean isFirst = true;
		if(stageIDs.get(instruction).size() == 0) {
			// Find back references
			HashSet<Integer> relatedStages = new HashSet<Integer>();
			for(Entry<Integer, HashSet<String>> kv : relatedInstructionsPerStage.entrySet()) {
				if(kv.getValue().contains(instruction)) {
					relatedStages.add(kv.getKey());
				}
			}
			HashSet<String> relatedInstructions = new HashSet<String>();
			for(Entry<String, Integer> kv : stageIDs.entries()) {
				if(relatedStages.contains(kv.getValue())) {
					relatedInstructions.add(kv.getKey());
				}
			}
			
			retVal.append(getInQuotes("backReferences") + ": [\n");
			boolean isFirstRelInst = true;
			for(String relInst : relatedInstructions) {
				if(!isFirstRelInst) {
					retVal.append(",\n");
				}
				retVal.append(getInQuotes(relInst));
				isFirstRelInst = false;
			}
			retVal.append("], \n");
		}
		else {
			retVal.append(getInQuotes("stages") + ": {");
			for(Integer stageId : stageIDs.get(instruction)) {
				String stageDAG = "";
				String stageTimeLine = "";
				
				if(getStageDAGs(stageId) != null) {
					stageDAG = getStageDAGs(stageId).toString();
				}
				
				if(getStageTimeLine(stageId) != null) {
					stageTimeLine = getStageTimeLine(stageId).toString();
				}
				
				long stageExecutionTime = stageExecutionTimes.get(stageId);
				if(!isFirst) {
					retVal.append(",\n");
				}
				
				retVal.append(getInQuotes("" + stageId) + ": {");
				
				// Now add related instructions
				HashSet<String> relatedInstructions = relatedInstructionsPerStage.get(stageId);
				
				retVal.append(getInQuotes("relatedInstructions") + ": [\n");
				boolean isFirstRelInst = true;
				for(String relInst : relatedInstructions) {
					if(!isFirstRelInst) {
						retVal.append(",\n");
					}
					retVal.append(getInQuotes(relInst));
					isFirstRelInst = false;
				}
				retVal.append("],\n");
				
				retVal.append(getInQuotes("DAG") + ":")
					  .append(
							getInQuotes(
							getEscapedJSON(stageDAG.replaceAll("toggleDagViz\\(false\\)", "toggleDagViz(false, this)")) 
							) + ",\n"
							)
					  .append(getInQuotes("stageExecutionTime") + ":" + stageExecutionTime + ",\n")
					  .append(getInQuotes("timeline") + ":")
					  .append(
							getInQuotes(
								getEscapedJSON(
								stageTimeLine
								.replaceAll("drawTaskAssignmentTimeline\\(", "registerTimelineData(" + stageId + ", ")
								.replaceAll("class=\"expand-task-assignment-timeline\"",  "class=\"expand-task-assignment-timeline\" onclick=\"toggleStageTimeline(this)\""))
								)
							 )
					  .append("}");
				
				isFirst = false;
			}
			retVal.append("}, ");
		}
		
		
		retVal.append(getInQuotes("jobs") + ": {");
		isFirst = true;
		for(Integer jobId : jobIDs.get(instruction)) {
			String jobDAG = "";
			
			if(getJobDAGs(jobId) != null) {
				jobDAG = getJobDAGs(jobId).toString();
			}
			if(!isFirst) {
				retVal.append(",\n");
			}
			
			retVal.append(getInQuotes("" + jobId) + ": {")
					.append(getInQuotes("DAG") + ":" ) 
					.append(getInQuotes(
						getEscapedJSON(jobDAG.replaceAll("toggleDagViz\\(true\\)", "toggleDagViz(true, this)")) 
						) + "}\n");
			
			isFirst = false;
		}
		retVal.append("}, ");
		
		return retVal.toString();
	}

	
	String [] dmlLines = null;
	private String getExpression(Location loc) {
		try {
			if(dmlLines == null) {
				dmlLines =  dmlStrForMonitoring.split("\\r?\\n");
			}
			if(loc.beginLine == loc.endLine) {
				return dmlLines[loc.beginLine-1].substring(loc.beginCol-1, loc.endCol);
			}
			else {
				String retVal = dmlLines[loc.beginLine-1].substring(loc.beginCol-1);
				for(int i = loc.beginLine+1; i < loc.endLine; i++) {
					retVal += "<br />" +  dmlLines[i-1];
				}
				retVal += "<br />" + dmlLines[loc.endLine-1].substring(0, loc.endCol);
				return retVal;
			}
		}
		catch(Exception e) {
			return null; // "[[" + loc.beginLine + "," + loc.endLine + "," + loc.beginCol + "," + loc.endCol + "]]";
		}
	}
	
	
	private String getExpressionInJSON(Location loc) {
		try {
			if(dmlLines == null) {
				dmlLines =  dmlStrForMonitoring.split("\\r?\\n");
			}
			if(loc.beginLine == loc.endLine) {
				return dmlLines[loc.beginLine-1].substring(loc.beginCol-1, loc.endCol);
			}
			else {
				String retVal = dmlLines[loc.beginLine-1].substring(loc.beginCol-1);
				for(int i = loc.beginLine+1; i < loc.endLine; i++) {
					retVal += "\\n" +  dmlLines[i-1];
				}
				retVal += "\\n" + dmlLines[loc.endLine-1].substring(0, loc.endCol);
				return retVal;
			}
		}
		catch(Exception e) {
			return null; // "[[" + loc.beginLine + "," + loc.endLine + "," + loc.beginCol + "," + loc.endCol + "]]";
		}
	}
	
	public Seq<Node> getStageDAGs(int stageIDs) {
		if(_sparkListener == null || _sparkListener.stageDAGs == null)
			return null;
		else
			return _sparkListener.stageDAGs.get(stageIDs);
	}
	
	public Long getStageExecutionTime(int stageID) {
		if(_sparkListener == null || _sparkListener.stageDAGs == null)
			return null;
		else
			return _sparkListener.stageExecutionTime.get(stageID);
	}
	
	public Seq<Node> getJobDAGs(int jobID) {
		if(_sparkListener == null || _sparkListener.jobDAGs == null)
			return null;
		else
			return _sparkListener.jobDAGs.get(jobID);
	}
	
	public Seq<Node> getStageTimeLine(int stageIDs) {
		if(_sparkListener == null || _sparkListener.stageTimeline == null)
			return null;
		else
			return _sparkListener.stageTimeline.get(stageIDs);
	}
	public void setLineageInfo(Instruction inst, String plan) {
		lineageInfo.put(getInstructionString(inst), plan);
	}
	public void setStageId(Instruction inst, int stageId) {
		stageIDs.put(getInstructionString(inst), stageId);
	}
	public void setJobId(Instruction inst, int jobId) {
		jobIDs.put(getInstructionString(inst), jobId);
	}
	public void setInstructionLocation(Location loc, Instruction inst) {
		String instStr = getInstructionString(inst);
		instructions.put(loc, instStr);
		instructionCreationTime.put(instStr, System.currentTimeMillis());
	}
	private String getInstructionString(Instruction inst) {
		String tmp = inst.toString();
		tmp = tmp.replaceAll(Lop.OPERAND_DELIMITOR, " ");
		tmp = tmp.replaceAll(Lop.DATATYPE_PREFIX, ".");
		tmp = tmp.replaceAll(Lop.INSTRUCTION_DELIMITOR, ", ");
		return tmp;
	}
}
