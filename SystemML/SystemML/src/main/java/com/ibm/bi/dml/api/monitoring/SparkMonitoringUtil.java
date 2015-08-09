package com.ibm.bi.dml.api.monitoring;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import scala.collection.Seq;
import scala.xml.Node;

import com.google.common.collect.Multimap;
import com.google.common.collect.TreeMultimap;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;
import com.ibm.bi.dml.runtime.instructions.spark.functions.SparkListener;

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
	
	
	private Multimap<Location, String> instructions = TreeMultimap.create();
	private Multimap<String, Integer> stageIDs = TreeMultimap.create();  // instruction -> stageIds
	private HashMap<String, String> lineageInfo = new HashMap<String, String>();	// instruction -> lineageInfo
	private HashMap<String, Long> instructionCreationTime = new HashMap<String, Long>();
	public Seq<Node> getStageDAGs(int stageIDs) {
		if(_sparkListener == null || _sparkListener.stageDAGs == null)
			return null;
		else
			return _sparkListener.stageDAGs.get(stageIDs);
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
