/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */


package com.ibm.bi.dml.runtime.instructions.spark.functions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.spark.JavaSparkListener;
import org.apache.spark.SparkContext;
import org.apache.spark.scheduler.SparkListenerExecutorMetricsUpdate;
import org.apache.spark.scheduler.SparkListenerStageCompleted;
import org.apache.spark.scheduler.SparkListenerStageSubmitted;
import org.apache.spark.ui.jobs.UIData.TaskUIData;

import scala.Option;
import scala.collection.Seq;
import scala.xml.Node;

import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLContextProxy;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;

/**
 * This class is only used by MLContext for now. It is used to provide UI data for Python notebook.
 *
 */
public class SparkListener extends JavaSparkListener {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	

	public SparkListener(SparkContext sc) {
		this._sc = sc;
	}
	// protected SparkExecutionContext sec = null;
	protected SparkContext _sc = null;
	protected Set<SPInstruction> currentInstructions = new HashSet<SPInstruction>();
	private HashMap<Integer, ArrayList<TaskUIData>> stageTaskMapping = new HashMap<Integer, ArrayList<TaskUIData>>();  
	
	public HashMap<Integer, Seq<Node>> stageDAGs = new HashMap<Integer, Seq<Node>>();
	public HashMap<Integer, Seq<Node>> stageTimeline = new HashMap<Integer, Seq<Node>>();
	
	public void addCurrentInstruction(SPInstruction inst) {
		synchronized(currentInstructions) {
			currentInstructions.add(inst);
		}
	}
	
	public void removeCurrentInstruction(SPInstruction inst) {
		synchronized(currentInstructions) {
			currentInstructions.remove(inst);
		}
	}

	@Override
	public void onExecutorMetricsUpdate(
			SparkListenerExecutorMetricsUpdate executorMetricsUpdate) {
		super.onExecutorMetricsUpdate(executorMetricsUpdate);
	}
	
	@Override
	public void onStageSubmitted(SparkListenerStageSubmitted stageSubmitted) {
		super.onStageSubmitted(stageSubmitted);
		// stageSubmitted.stageInfo()
		
		Integer stageID = stageSubmitted.stageInfo().stageId();
		synchronized(currentInstructions) {
			stageTaskMapping.put(stageID, new ArrayList<TaskUIData>());
		}
		
		Option<org.apache.spark.ui.scope.RDDOperationGraph> rddOpGraph = Option.apply(org.apache.spark.ui.scope.RDDOperationGraph.makeOperationGraph(stageSubmitted.stageInfo()));
		Seq<Node> stageDAG = org.apache.spark.ui.UIUtils.showDagVizForStage(stageID, rddOpGraph);
		// Use org.apache.spark.ui.jobs.StagePage, org.apache.spark.ui.jobs.JobPage's makeTimeline method() to print timeline
		
		stageDAGs.put(stageID, stageDAG);
		// Seq<RDDInfo> rddsInvolved = stageSubmitted.stageInfo().rddInfos();
		
		synchronized(currentInstructions) {
			for(SPInstruction inst : currentInstructions) {
				MLContext mlContext = MLContextProxy.getActiveMLContext();
				if(mlContext != null && mlContext.getMonitoringUtil() != null) {
					mlContext.getMonitoringUtil().setStageId(inst, stageSubmitted.stageInfo().stageId());
				}
			}
		}
	}
	
	@Override
	public void onTaskEnd(org.apache.spark.scheduler.SparkListenerTaskEnd taskEnd) {
		Integer stageID = taskEnd.stageId();
		
		synchronized(currentInstructions) {
			if(stageTaskMapping.containsKey(stageID)) {
				Option<String> errorMessage = Option.apply(null); // TODO
				TaskUIData taskData = new TaskUIData(taskEnd.taskInfo(), Option.apply(taskEnd.taskMetrics()), errorMessage);
				stageTaskMapping.get(stageID).add(taskData);
			}
			else {
				// TODO: throw exception
			}
		}
	};
	
	@Override
	public void onStageCompleted(SparkListenerStageCompleted stageCompleted) {
		super.onStageCompleted(stageCompleted);	
	}
	
}
