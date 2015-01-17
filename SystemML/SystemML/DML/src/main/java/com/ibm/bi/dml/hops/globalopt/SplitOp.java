/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.HopsVisitor.Flag;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;

/**
 * Counter part of a Merge Node
 */
public class SplitOp extends CrossBlockOp 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Hop secondOutput;
	private MergeOp mergeNode;
	
	public SplitOp(Hop leftInput, Hop output) {
		super(leftInput, output);
	}
	
	public void addOutput(Hop output) {
		if(this.output == null)
		{
			this.output = output;
			return;
		}
		if(this.secondOutput == null) {
			this.secondOutput = output;
			return;
		}
		throw new RuntimeException("This SplitNode: " + getName() + " already has two outputs assigned!"); 
	}

	public Hop getSecondOutput() {
		return secondOutput;
	}

	public MergeOp getMergeNode() {
		return mergeNode;
	}

	public void setMergeNode(MergeOp mergeNode) {
		this.mergeNode = mergeNode;
	}
	
	@Override
	public void refreshSizeInformation() {
		this._dim1 = leftInput.getDim1();
		this._dim2 = leftInput.getDim2();
	}

	@Override
	public void accept(HopsVisitor visitor) {
		if(visitor.traverseBackwards()) {
			acceptSinkToSource(visitor);
		}else{
			acceptSourceToSink(visitor);
		}
	}

	private void acceptSinkToSource(HopsVisitor visitor)
	{
		throw new RuntimeException("Not implemented yet!");
	}
	
	/**
	 * @param visitor
	 */
	private void acceptSourceToSink(HopsVisitor visitor) {
		Flag flag = visitor.preVisit(this);
		this.hopsVisited.put(visitor, true);
		
		if(flag != Flag.STOP_INPUT 
				&& this.leftInput != null 
				&& !this.leftInput.isHopsVisited(visitor)
				&& visitor.matchesPattern(this.leftInput)) {
			this.leftInput.accept(visitor);
		}
		
		visitor.visit(this);
		
		if(flag != Flag.STOP_OUTPUT) { 
				if(this.output != null 
						&& !this.output.isHopsVisited(visitor)
						&& visitor.matchesPattern(this.output)) {
					this.output.accept(visitor);
				}
				
				if(this.secondOutput != null 
						&& !this.secondOutput.isHopsVisited(visitor)
						&& visitor.matchesPattern(this.secondOutput)) {
					this.secondOutput.accept(visitor);
				}
		}
		
		visitor.postVisit(this);
	}
	
	@Override
	public Lop constructLops() throws HopsException, LopsException {
		if(getLops() != null) {
			return getLops();
		}
		else {
			SplitLop lop = new SplitLop();
			setLops(lop);
			return lop;
		}
	}
	
}
