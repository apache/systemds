/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

public class Summary
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private double _costsInitial = -1;
	private double _costsOptimal = -1;
	private long _numEnumPlans = -1;
	private long _numPrunedInvalidPlans = -1;
	private long _numPrunedSuboptPlans  = -1;
	private long _numCompiledPlans = -1;
	private long _numCostedPlans = -1;
	private long _numEnumPlanMismatch = -1;
	private long _numFinalPlanMismatch = -1;
	private double _timeGDFGraph = -1;
	private double _timeOptim = -1;

	public void setCostsInitial(double costsInitial) {
		_costsInitial = costsInitial;
	}

	public void setCostsOptimal(double costsOptimal) {
		_costsOptimal = costsOptimal;
	}

	public void setNumEnumPlans(long numEnumPlans) {
		_numEnumPlans = numEnumPlans;
	}

	public void setNumPrunedInvalidPlans(long numPrunedInvalidPlans) {
		_numPrunedInvalidPlans = numPrunedInvalidPlans;
	}

	public void setNumPrunedSuboptPlans(long numPrunedSuboptPlans) {
		_numPrunedSuboptPlans = numPrunedSuboptPlans;
	}

	public void setNumCompiledPlans(long numCompiledPlans) {
		_numCompiledPlans = numCompiledPlans;
	}

	public void setNumCostedPlans(long numCostedPlans) {
		_numCostedPlans = numCostedPlans;
	}

	public void setNumEnumPlanMismatch(long numEnumPlanMismatch) {
		_numEnumPlanMismatch = numEnumPlanMismatch;
	}

	public void setNumFinalPlanMismatch(long numFinalPlanMismatch) {
		_numFinalPlanMismatch = numFinalPlanMismatch;
	}

	public void setTimeGDFGraph(double timeGDFGraph) {
		_timeGDFGraph = timeGDFGraph;
	}

	public void setTimeOptim(double timeOptim) {
		_timeOptim = timeOptim;
	}

	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();

		sb.append("\nGlobal Optimization Summary:\n");
		sb.append("-- costs of initial plan:  "+_costsInitial +"\n");
		sb.append("-- costs of optimal plan:  "+_costsOptimal +"\n");
		sb.append("-- # enumerated plans:     "+_numEnumPlans +"\n");
		sb.append("-- # pruned invalid plans: "+_numPrunedInvalidPlans +"\n");
		sb.append("-- # pruned subopt plans:  "+_numPrunedSuboptPlans +"\n");
		sb.append("-- # program compilations: "+_numCompiledPlans +"\n");
		sb.append("-- # program costings:     "+_numCostedPlans +"\n");
		sb.append("-- # enum plan mismatch:   "+_numEnumPlanMismatch +"\n");
		sb.append("-- # final plan mismatch:  "+_numFinalPlanMismatch +"\n");
		sb.append("-- graph creation time:    "+String.format("%.3f", _timeGDFGraph/1000)+" sec.\n");
		sb.append("-- optimization time:      "+String.format("%.3f", _timeOptim/1000)+" sec.");
	
		return sb.toString();
	}
}
