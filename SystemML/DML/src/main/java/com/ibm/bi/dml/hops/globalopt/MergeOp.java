/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.ArrayList;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.HopsVisitor.Flag;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;

/**
 *
 */
public class MergeOp extends CrossBlockOp 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	protected Hop rightInput;
	protected long leftDim1, leftDim2, rightDim1, rightDim2;

	public MergeOp(Hop leftInput, Hop output) {
		super(leftInput, output);
	}

	public MergeOp(Hop leftInput, Hop rightInput, Hop output) {
		super(leftInput, output);
		//redundant but more explicit
		this.leftDim1 = leftInput.get_dim1();
		this.rightDim2 = leftInput.get_dim2();
		this.rightInput = rightInput;
		this.rightDim1 = rightInput.get_dim1();
		this.rightDim2 = rightInput.get_dim2();
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
		//TODO: implement me
		throw new RuntimeException("Not implemented yet!");
	}
	
	/**
	 * Involves a lot of stupid code duplication (which probably still causes bugs)
	 * TODO: ultimately find a way to remove that!
	 */
	private void acceptSourceToSink(HopsVisitor visitor) {
		Flag flag = visitor.preVisit(this);
		super.hopsVisited.put(visitor, true);
		if (flag != Flag.STOP_INPUT) {
			if (this.leftInput != null 
					&& !this.leftInput.isHopsVisited(visitor) 
					&& visitor.matchesPattern(this.leftInput)) {
				this.leftInput.accept(visitor);
			}
			if (this.rightInput != null
					&& !this.rightInput.isHopsVisited(visitor)
					&& visitor.matchesPattern(this.rightInput)) {
				this.rightInput.accept(visitor);
			}
		}
		
		visitor.visit(this);
		
		if(flag != Flag.STOP_OUTPUT 
				&& this.output != null
				&& !this.output.isHopsVisited(visitor)
				&& visitor.matchesPattern(this.output)) {
			this.output.accept(visitor);
		}
		
		visitor.postVisit(this);
	}

	@Override
	public long get_dim1() {
		return Math.max(this.leftDim1, this.rightDim1);
	}

	@Override
	public long get_dim2() {
		return Math.max(this.leftDim2, this.rightDim2);
	}
	
	public Hop getRightInput() {
		return rightInput;
	}
	
	@Override
	public ArrayList<Hop> getInput() {
		ArrayList<Hop> retVal = super.getInput();
		retVal.add(this.rightInput);
		return retVal;
	}
	
	@Override
	public void setCrossBlockOutput(CrossBlockOp crossBlockOutput) {
		this.output = crossBlockOutput;
	}

	/*
	@Override
	public void propagateBlockSize() {
		super.propagateBlockSize();
		long rowsPerBlock = this.get_rows_in_block();
		long colsPerBlock = this.get_cols_in_block();
		Hop rightInput = this.getRightInput();
		if (rightInput != null) {
			rightInput.set_cols_in_block(colsPerBlock);
			rightInput.set_rows_in_block(rowsPerBlock);
		}
	}
	
	*/
	
	@Override
	public void refreshSizeInformation() {
		this.leftDim1 = leftInput.get_dim1();
		this.leftDim2 = leftInput.get_dim2();
		
		this.rightDim1 = rightInput.get_dim1();
		this.rightDim2 = rightInput.get_dim2();
	}
	
	public Lop constructLops() throws HopsException, LopsException {
		if(get_lops() != null) {
			return get_lops();
		} else {
			MergeLop mergeLop = new MergeLop();
			set_lops(mergeLop);
			
			return mergeLop;
		}
	}
	
}
