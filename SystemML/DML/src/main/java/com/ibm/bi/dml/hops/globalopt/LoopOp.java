/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.sql.sqllops.SQLLops;

/**
 * Represents a loop in a maximal global graph.
 * TODO: Let's see if this is enough or if subclasses are needed.
 * 
 * Preliminary answer: probably not, since we need multiple, semantically different outputs
 */
public class LoopOp extends CrossBlockOp 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<String, HopsDag> internalGraphs;
	private Map<String, Hop> loopInputs;
	private Map<String, Hop> loopOutputs;
	
	public LoopOp(Map<String, HopsDag> internalGraphs) {
		this.internalGraphs = new HashMap<String, HopsDag>();
		this.loopInputs = new HashMap<String, Hop>();
		this.loopOutputs = new HashMap<String, Hop>(); 
		this.setInternalGraphs(internalGraphs);
	}

	public enum LoopType {
		WHILE, FOR
	}
	
	private boolean knownNumberOfExecutions;
	private LoopType loopType;
	
	private Hop predicate;
	
	@Override
	public boolean allowsAllExecTypes() {
		return false;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#clone()
	 */
	@Override
	public Object clone() throws CloneNotSupportedException {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#constructLops()
	 */
	@Override
	public Lop constructLops() throws HopsException, LopsException {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#constructSQLLOPs()
	 */
	@Override
	public SQLLops constructSQLLOPs() {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#getOpString()
	 */
	@Override
	public String getOpString() {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#optFindExecType()
	 */
	@Override
	protected ExecType optFindExecType() {
		// TODO Auto-generated method stub
		return null;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.hops.Hops#refreshSizeInformation()
	 */
	@Override
	public void refreshSizeInformation() {
		// TODO Auto-generated method stub

	}

	public boolean isKnownNumberOfExecutions() {
		return knownNumberOfExecutions;
	}

	public void setKnownNumberOfExecutions(boolean knownNumberOfExecutions) {
		this.knownNumberOfExecutions = knownNumberOfExecutions;
	}

	public LoopType getLoopType() {
		return loopType;
	}

	public void setLoopType(LoopType loopType) {
		this.loopType = loopType;
	}

	public Hop getPredicate() {
		return predicate;
	}

	public void setPredicate(Hop predicate) {
		this.predicate = predicate;
	}

	public Map<String, HopsDag> getInternalGraphs() {
		return internalGraphs;
	}

	public void setInternalGraphs(Map<String, HopsDag> internalGraphs) {
		this.internalGraphs = internalGraphs;
		
		for(Entry<String, HopsDag> e : internalGraphs.entrySet()) {
			HopsDag dag = e.getValue();
			this.loopInputs.putAll(dag.getDagInputs());
			this.loopOutputs.putAll(dag.getDagOutputs());
		}
		
	}

	public void addInternalGraph(String name, HopsDag graph) {
		this.internalGraphs.put(name, graph);
	}

	public Map<String, Hop> getLoopInputs() {
		return loopInputs;
	}

	public void setLoopInputs(Map<String, Hop> loopInputs) {
		this.loopInputs = loopInputs;
	}
	
	public void addLoopInput(Hop input) {
		this.loopInputs.put(input.get_name(), input);
	}

	public Map<String, Hop> getLoopOutputs() {
		return loopOutputs;
	}

	public void setLoopOutputs(Map<String, Hop> loopOutput) {
		this.loopOutputs = loopOutput;
	}
	
	public void addLoopOutput(Hop output) {
		this.loopOutputs.put(output.get_name(), output);
	}
	
	@Override
	public void accept(HopsVisitor visitor) {
		// TODO Auto-generated method stub
		super.accept(visitor);
	}
	
	/**
	 * Detects whether this loop body is loop insensitive. It does so by examining the dimensions of 
	 * corresponding input and output matrices.
	 * @return
	 */
	public boolean isBodyLoopInsensitive() {
		boolean retVal = true;
		
		for(String inputVarName : this.loopInputs.keySet()) {
			Hop input = this.loopInputs.get(inputVarName);
			Hop output = this.loopOutputs.get(inputVarName);
			if(output != null) {
				if(input.get_dim1() != output.get_dim1() || input.get_dim2() != output.get_dim2()) {
					retVal = false;
					break;
				}
			}
		}
		
		return retVal;
	}
}
