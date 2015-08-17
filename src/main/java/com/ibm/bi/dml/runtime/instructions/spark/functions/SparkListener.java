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

import org.apache.spark.SparkContext;
import org.apache.spark.scheduler.SparkListenerExecutorMetricsUpdate;
import org.apache.spark.scheduler.SparkListenerStageCompleted;
import org.apache.spark.scheduler.SparkListenerStageSubmitted;
import org.apache.spark.storage.RDDInfo;
import org.apache.spark.ui.jobs.StagesTab;
import org.apache.spark.ui.jobs.UIData.TaskUIData;
import org.apache.spark.ui.scope.RDDOperationGraphListener;

import scala.Option;
import scala.collection.Iterator;
import scala.collection.Seq;
import scala.xml.Node;

import com.ibm.bi.dml.api.MLContext;
import com.ibm.bi.dml.api.MLContextProxy;
import com.ibm.bi.dml.runtime.instructions.spark.SPInstruction;

// Instead of extending org.apache.spark.JavaSparkListener
/**
 * This class is only used by MLContext for now. It is used to provide UI data for Python notebook.
 *
 */
public class SparkListener extends RDDOperationGraphListener {
	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public SparkListener(SparkContext sc) {
		super(sc.conf());
		this._sc = sc;
	}
	
	// protected SparkExecutionContext sec = null;
	protected SparkContext _sc = null;
	protected Set<SPInstruction> currentInstructions = new HashSet<SPInstruction>();
	private HashMap<Integer, ArrayList<TaskUIData>> stageTaskMapping = new HashMap<Integer, ArrayList<TaskUIData>>();  
	
	public HashMap<Integer, Seq<Node>> stageDAGs = new HashMap<Integer, Seq<Node>>();
	public HashMap<Integer, Seq<Node>> stageTimeline = new HashMap<Integer, Seq<Node>>();
	public HashMap<Integer, Seq<Node>> jobDAGs = new HashMap<Integer, Seq<Node>>();
	public HashMap<Integer, Long> stageExecutionTime = new HashMap<Integer, Long>();
	public HashMap<Integer, ArrayList<Integer>> stageRDDMapping = new HashMap<Integer, ArrayList<Integer>>(); 
	
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
	public void onJobEnd(org.apache.spark.scheduler.SparkListenerJobEnd jobEnd) {
		super.onJobEnd(jobEnd);
		int jobID = jobEnd.jobId();
		Seq<Node> jobNodes = org.apache.spark.ui.UIUtils.showDagVizForJob(jobID, this.getOperationGraphForJob(jobID));
		jobDAGs.put(jobID, jobNodes);
		synchronized(currentInstructions) {
			for(SPInstruction inst : currentInstructions) {
				MLContext mlContext = MLContextProxy.getActiveMLContext();
				if(mlContext != null && mlContext.getMonitoringUtil() != null) {
					mlContext.getMonitoringUtil().setJobId(inst, jobID);
				}
			}
		}
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
		
		Iterator<RDDInfo> iter = stageSubmitted.stageInfo().rddInfos().toList().toIterator();
		ArrayList<Integer> rddIDs = new ArrayList<Integer>();
		while(iter.hasNext()) {
			RDDInfo rddInfo = iter.next();
			rddIDs.add(rddInfo.id());
		}
		stageRDDMapping.put(stageSubmitted.stageInfo().stageId(), rddIDs);
		
		
		Seq<Node> stageDAG = org.apache.spark.ui.UIUtils.showDagVizForStage(stageID, rddOpGraph);
		stageDAGs.put(stageID, stageDAG);
		
		// Use org.apache.spark.ui.jobs.StagePage, org.apache.spark.ui.jobs.JobPage's makeTimeline method() to print timeline
//		try {
			ArrayList<TaskUIData> taskUIData = stageTaskMapping.get(stageID);
			Seq<Node> currentStageTimeline = (new org.apache.spark.ui.jobs.StagePage(new StagesTab(_sc.ui().get())))
					.makeTimeline(
							scala.collection.JavaConversions.asScalaBuffer(taskUIData).toList(), 
					System.currentTimeMillis());
			stageTimeline.put(stageID, currentStageTimeline);
//		}
//		catch(Exception e) {} // Ignore 
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
		try {
			long completionTime = Long.parseLong(stageCompleted.stageInfo().completionTime().get().toString());
			long submissionTime = Long.parseLong(stageCompleted.stageInfo().submissionTime().get().toString());
			stageExecutionTime.put(stageCompleted.stageInfo().stageId(), completionTime-submissionTime);
		}
		catch(Exception e) {}
	}
	
}
