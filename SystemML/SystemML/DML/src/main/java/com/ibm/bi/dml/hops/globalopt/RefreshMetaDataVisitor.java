/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt;

import java.util.HashMap;
import java.util.Map;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;

/**
 * Creates and maintains a directory of {@link HopsMetaData} objects for each node in the traversed {@link HopsDag}.
 * 
 */
public class RefreshMetaDataVisitor implements HopsVisitor 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Map<Hop, HopsMetaData> metaDataDirectory = new HashMap<Hop, HopsMetaData>();
	
	@Override
	public boolean matchesPattern(Hop hops) {
		return true;
	}

	@Override
	public Flag postVisit(Hop hops) {
		return Flag.GO_ON;
	}
	
	/**
	 * Safe old dimension info, update dimension info using the new global structure, extract the data 
	 * and put the old values back into hops parameter. This way, we don't have to replicate the internal logic 
	 * of refreshSizeInformation() and still do not interfere with the existing compilation/optimization process.
	 */
	@Override
	public Flag visit(Hop hops) {
		if(hops instanceof LiteralOp)
			return Flag.GO_ON;
		
		long dim1Before = hops.getDim1();
		long dim2Before = hops.getDim2();
		long nnzBefore = hops.getNnz();
		long dim1LeftBefore = -1L;
		long dim2LeftBefore = -1L;
		long dim1RightBefore = -1L;
		long dim2RightBefore = -1L;
		
		//save values and set meta data
		if(hops instanceof DataOp && (((DataOp)hops).get_dataop() == DataOpTypes.TRANSIENTREAD)) {
			Hop crossBlockInput = hops.getCrossBlockInput();
			dim1LeftBefore = crossBlockInput.getDim1();
			dim2LeftBefore = crossBlockInput.getDim2();
			HopsMetaData hopsMetaData = this.metaDataDirectory.get(crossBlockInput);
			if(hopsMetaData != null) {
				crossBlockInput.setDim1(hopsMetaData.getRows());
				crossBlockInput.setDim2(hopsMetaData.getCols());
			}
		}else 
		if(hops.getInput() != null ){
			if(!hops.getInput().isEmpty() ) {
				Hop leftInput = hops.getInput().get(0);
				dim1LeftBefore = leftInput.getDim1();
				dim2LeftBefore = leftInput.getDim2();
				HopsMetaData hopsMetaData = this.metaDataDirectory.get(leftInput);
				if(hopsMetaData != null) {
					leftInput.setDim1(hopsMetaData.getRows());
					leftInput.setDim2(hopsMetaData.getCols());
				}
			}
			if(hops.getInput().size() > 1) {
				Hop rightInput = hops.getInput().get(1);
				dim1RightBefore = rightInput.getDim1();
				dim2RightBefore = rightInput.getDim2();
				HopsMetaData hopsMetaData = this.metaDataDirectory.get(rightInput);
				if(hopsMetaData != null) {
					rightInput.setDim1(hopsMetaData.getRows());
					rightInput.setDim2(hopsMetaData.getCols());
				}
			}
		}
		//unfortunately, this needs to be handled separately
		if(hops instanceof MergeOp) {
			MergeOp merge = (MergeOp)hops;
			Hop leftInput = merge.getLeftInput();
			dim1LeftBefore = leftInput.getDim1();
			dim2LeftBefore = leftInput.getDim2();
			HopsMetaData hopsMetaData = this.metaDataDirectory.get(leftInput);
			if(hopsMetaData != null) {
				leftInput.setDim1(hopsMetaData.getRows());
				leftInput.setDim2(hopsMetaData.getCols());
			}
			Hop rightInput = merge.getRightInput();
			dim1RightBefore = rightInput.getDim1();
			dim2RightBefore = rightInput.getDim2();
			HopsMetaData hopsMetaDataR = this.metaDataDirectory.get(rightInput);
			if(hopsMetaDataR != null) {
				rightInput.setDim1(hopsMetaDataR.getRows());
				rightInput.setDim2(hopsMetaDataR.getCols());
			}
		}else if(hops instanceof CrossBlockOp) {
			CrossBlockOp merge = (CrossBlockOp)hops;
			Hop leftInput = merge.getLeftInput();
			dim1LeftBefore = leftInput.getDim1();
			dim2LeftBefore = leftInput.getDim2();
			HopsMetaData hopsMetaData = this.metaDataDirectory.get(leftInput);
			if(hopsMetaData != null) {
				leftInput.setDim1(hopsMetaData.getRows());
				leftInput.setDim2(hopsMetaData.getCols());
			}
		}
		
		hops.refreshSizeInformation();
		
		HopsMetaData meta = this.metaDataDirectory.get(hops);
		if(meta == null)
		{
			meta = new HopsMetaData(hops);
		} 
		meta.setRows(hops.getDim1());
		meta.setCols(hops.getDim2());
		meta.setNnz(hops.getNnz());
		meta.setColsInBlock(hops.getColsInBlock());
		meta.setRowsInBlock(hops.getRowsInBlock());
		
		//reset values
		if(hops instanceof DataOp && (((DataOp)hops).get_dataop() == DataOpTypes.TRANSIENTREAD)) {
			Hop crossBlockInput = hops.getCrossBlockInput();
			crossBlockInput.setDim1(dim1LeftBefore);
			crossBlockInput.setDim2(dim2LeftBefore);
		}else 
		if(hops.getInput() != null ){
			if(!hops.getInput().isEmpty()) {
				Hop leftInput = hops.getInput().get(0);
				leftInput.setDim1(dim1LeftBefore);
				leftInput.setDim2(dim2LeftBefore);
			}
			if(hops.getInput().size() > 1) {
				Hop rightInput = hops.getInput().get(1);
				rightInput.setDim1(dim1RightBefore);
				rightInput.setDim2(dim2RightBefore);
			}
		}
		//unfortunately, this needs to be handled separately
		if(hops instanceof MergeOp) {
			MergeOp merge = (MergeOp)hops;
			Hop leftInput = merge.getLeftInput();
			leftInput.setDim1(dim1LeftBefore);
			leftInput.setDim2(dim2LeftBefore);
			Hop rightInput = merge.getRightInput();
			rightInput.setDim1(dim1RightBefore );
			rightInput.setDim2(dim2RightBefore );
		}else if(hops instanceof CrossBlockOp) {
			CrossBlockOp merge = (CrossBlockOp)hops;
			Hop leftInput = merge.getLeftInput();
			leftInput.setDim1(dim1LeftBefore);
			leftInput.setDim2(dim2LeftBefore);
		}
		
		hops.setDim1(dim1Before);
		hops.setDim2(dim2Before);
		hops.setNnz(nnzBefore);
		
		this.metaDataDirectory.put(hops, meta);
		hops.setMetaData(meta);
		return Flag.GO_ON;
	}
	

	@Override
	public Flag preVisit(Hop hops) {
		return Flag.GO_ON;
	}

	public void initialize(HopsDag dag) {
		this.metaDataDirectory = new HashMap<Hop, HopsMetaData>();
		Map<String, Hop> dagInputs = dag.getDagInputs();
		for(Hop h : dagInputs.values()) {
			h.accept(this);
		}
		
		dag.setMetaDataDirectory(this.metaDataDirectory);
	}

	public HopsMetaData getMetaData(Hop hops) {
		return this.metaDataDirectory.get(hops);
	}

	@Override
	public boolean traverseBackwards() {
		return false;
	}

}
