/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.transform;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.AggBinaryOp;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.ReblockOp;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.hops.globalopt.HopsDag;
import com.ibm.bi.dml.hops.globalopt.MergeOp;
import com.ibm.bi.dml.hops.globalopt.RewriteRule;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;

/**
 * Rewrite is very specific to TSMM use.
 * TODO: extend
 */
public class BlockSizeRule extends RewriteRule
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(BlockSizeRule.class); 
	
	private Set<String> patternSet = new HashSet<String>();
	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.HopsVisitor#matchesPattern(com.ibm.bi.dml.hops.Hops)
	 */
	@Override
	public boolean matchesPattern(Hop hops) {
		if(hops.get_name() != null && this.patternSet.contains(hops.get_name()))
			return true;
		return false;
	}

	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.HopsVisitor#visit(com.ibm.bi.dml.hops.Hops)
	 */
	@Override
	public Flag preVisit(Hop hops) {
		updateRewritePath(hops);
		
		if(LOG.isDebugEnabled()) {
			LOG.debug(hops.getMetaData());
		}
		
		if(hops instanceof ReblockOp) {
			return Flag.STOP_INPUT;
		}
		
		propageBlockSize(hops);
		
		return Flag.GO_ON;
	}

	/**
	 * Propagate the block size info in the meta data objects.
	 * TODO: add apply switch
	 * @param hops
	 */
	private void propageBlockSize(Hop hops) {
		
		HopsMetaData metaData = hops.getMetaData();
		
		ArrayList<Hop> inputList = hops.getInput();
		if(hops instanceof ReblockOp) {
			return;
		}
		
		if(hops instanceof DataOp && ((DataOp)hops).get_dataop() == DataOpTypes.PERSISTENTREAD) {
			ArrayList<Hop> parents = hops.getParent();
			if(parents != null && parents.size() > 0) { //this has to be true otherwise this hops has no purpose
				Hop firstParent = parents.get(0);
				if(!(firstParent instanceof ReblockOp)) {
					HopsMetaData parentMeta = firstParent.getMetaData();
					long parentRowsPerBlock = parentMeta.getRowsInBlock();
					long parentColsPerBlock = parentMeta.getColsInBlock();
					if(hops.get_rows_in_block() != parentRowsPerBlock 
							|| hops.get_cols_in_block() != parentColsPerBlock) 
					{
						metaData.setReblockRequired(true);
						metaData.setColsInBlock(parentColsPerBlock);
						metaData.setRowsInBlock(parentRowsPerBlock);
						this.changes.add(metaData);
					}
				}
			}
		}
		
		if(hops instanceof DataOp && ((DataOp)hops).get_dataop() == DataOpTypes.TRANSIENTREAD) {
			CrossBlockOp crossBlockInput = hops.getCrossBlockInput();
			if(!crossBlockInput.isHopsVisited(this)) {
				HopsMetaData crossMeta = crossBlockInput.getMetaData();
				crossMeta.setColsInBlock(metaData.getColsInBlock());
				crossMeta.setRowsInBlock(metaData.getRowsInBlock());
				//TODO: make this change application better!
				this.changes.add(crossMeta);
			}
			if(hops.getParent() != null )
			{
				for(Hop h : hops.getParent()) {
					HopsMetaData parentMeta = h.getMetaData();
					parentMeta.setRowsInBlock(metaData.getRowsInBlock());
					parentMeta.setColsInBlock(metaData.getColsInBlock());
					this.changes.add(parentMeta);
				}
			}
			
		}else if(hops instanceof DataOp && ((DataOp)hops).get_dataop() == DataOpTypes.TRANSIENTWRITE) {
			if(inputList.size() > 0 && !(inputList.get(0) instanceof LiteralOp)) {
				Hop leftInput = inputList.get(0);
				//TODO: change into maximum decision 
				if(!leftInput.isHopsVisited(this))
				{
					HopsMetaData leftMeta = leftInput.getMetaData();
					leftMeta.setColsInBlock(metaData.getColsInBlock());
					leftMeta.setRowsInBlock(metaData.getRowsInBlock());
					this.changes.add(leftMeta);
				}
			}
			CrossBlockOp crossBlockOutput = hops.getCrossBlockOutput();
			if(!crossBlockOutput.isHopsVisited(this)) {
				HopsMetaData crossMeta = crossBlockOutput.getMetaData();
				crossMeta.setColsInBlock(metaData.getColsInBlock());
				crossMeta.setRowsInBlock(metaData.getRowsInBlock());
				this.changes.add(crossMeta);
			}
		}else if(inputList != null) 
		{
			if(inputList.size() > 0 && !(inputList.get(0) instanceof LiteralOp)) {
				Hop leftInput = inputList.get(0);
				//TODO: change into maximum decision 
				if(!leftInput.isHopsVisited(this))
				{
					HopsMetaData leftMeta = leftInput.getMetaData();
					leftMeta.setColsInBlock(metaData.getColsInBlock());
					leftMeta.setRowsInBlock(metaData.getRowsInBlock());
					this.changes.add(leftMeta);
				}
			}
			if(inputList.size() > 1 && !(inputList.get(1) instanceof LiteralOp)) {
				Hop rightInput = inputList.get(1);
				if(!rightInput.isHopsVisited(this)){
					HopsMetaData rightMeta = rightInput.getMetaData();
					rightMeta.setColsInBlock(metaData.getColsInBlock());
					rightMeta.setRowsInBlock(metaData.getRowsInBlock());
					this.changes.add(rightMeta);
				}
			}
			
			if(hops.getParent() != null )
			{
				for(Hop h : hops.getParent()) {
					HopsMetaData parentMeta = h.getMetaData();
					parentMeta.setRowsInBlock(metaData.getRowsInBlock());
					parentMeta.setColsInBlock(metaData.getColsInBlock());
					this.changes.add(parentMeta);
				}
			}
		}
		if(hops instanceof MergeOp) {
			MergeOp mergeN = (MergeOp)hops;
			Hop leftInput = mergeN.getLeftInput();
			if(leftInput != null) {
				HopsMetaData leftMeta = leftInput.getMetaData();
				leftMeta.setColsInBlock(metaData.getColsInBlock());
				leftMeta.setRowsInBlock(metaData.getRowsInBlock());
				this.changes.add(leftMeta);
			}
			Hop rightInput = mergeN.getRightInput();
			if(rightInput != null) {
				HopsMetaData rightMeta = rightInput.getMetaData();
				rightMeta.setRowsInBlock(metaData.getRowsInBlock());
				rightMeta.setColsInBlock(metaData.getColsInBlock());
				this.changes.add(rightMeta);
			}
			Hop output = mergeN.getOutput();
			if(!output.isHopsVisited(this)) {
				HopsMetaData outMeta = output.getMetaData();
				outMeta.setRowsInBlock(metaData.getRowsInBlock());
				outMeta.setColsInBlock(metaData.getColsInBlock());
				this.changes.add(outMeta);
			}
		} if(hops instanceof CrossBlockOp) {
			CrossBlockOp crossNode = (CrossBlockOp)hops;
			Hop leftInput = crossNode.getLeftInput();
			if(leftInput != null) {
				HopsMetaData leftMeta = leftInput.getMetaData();
				leftMeta.setColsInBlock(metaData.getColsInBlock());
				leftMeta.setRowsInBlock(metaData.getRowsInBlock());
				this.changes.add(leftMeta);
			}
			
			Hop output = crossNode.getOutput();
			if(output != null && !output.isHopsVisited(this)) {
				HopsMetaData outMeta = output.getMetaData();
				outMeta.setRowsInBlock(metaData.getRowsInBlock());
				outMeta.setColsInBlock(metaData.getColsInBlock());
				this.changes.add(outMeta);
			}
			
		}
	}

	/**
	 * If {@link Hops} param is of the proper type, take both inputs and its output 
	 * @param hops
	 */
	private void updateRewritePath(Hop hops) {
		if(hops instanceof AggBinaryOp){
			AggBinaryOp aggOp = (AggBinaryOp)hops;
			if(aggOp.isMatrixMultiply())
			{
				Hop leftInput = aggOp.getInput().get(0);
				this.patternSet.add(leftInput.get_name());
				Hop rightInput = aggOp.getInput().get(1);
				this.patternSet.add(rightInput.get_name());
			}
		}
		if(hops instanceof BinaryOp) {
			Hop leftInput = hops.getInput().get(0);
			this.patternSet.add(leftInput.get_name());
			Hop rightInput = hops.getInput().get(1);
			this.patternSet.add(rightInput.get_name());
		}
		
		for(Hop p : hops.getParent()) {
			this.patternSet.add(p.get_name());
		}
		
		if(hops instanceof ReblockOp) {
			return;
		}
		
		if(hops instanceof FunctionOp) {
			FunctionOp funcOp = (FunctionOp)hops;
			this.patternSet.add(funcOp.get_name());
		}
	}

	/**
	 * 
	 */
	@Override
	public HopsDag rewrite(HopsDag toModify) {
		Map<String, Set<Hop>> hopsDirectory = toModify.getHopsDirectory();
		Set<Hop> aggBinOps = hopsDirectory.get(AggBinaryOp.class.getCanonicalName());
		
		if(aggBinOps == null || aggBinOps.isEmpty())
			return toModify;
			
		for(Hop mult : aggBinOps) {
			AggBinaryOp binOp = (AggBinaryOp)mult;
			HopsMetaData binMeta = binOp.getMetaData();
			if(binOp.isMatrixMultiply()) {
				MMTSJType checkTransposeSelf = binOp.checkTransposeSelf();
				if(checkTransposeSelf != MMTSJType.NONE) {
					if(checkTransposeSelf == MMTSJType.RIGHT) {
						Hop leftInput = binOp.getInput().get(0);
						HopsMetaData leftMetaData = leftInput.getMetaData();
						long rowsInBlock = leftMetaData.getRowsInBlock();
						long rows = leftMetaData.getRows();
						if(rows > rowsInBlock) {
							binMeta.setColsInBlock(rows);
							binMeta.setRowsInBlock(rows);
							binOp.accept(this);
						}
					}else {
						Hop rightInput = binOp.getInput().get(1);
						HopsMetaData rightMetaData = rightInput.getMetaData();
						long colsInBlock = rightMetaData.getColsInBlock();
						long cols = rightMetaData.getCols();
						if(cols > colsInBlock) {
							//do the necessary thing to fit cols into cols in blocks 
							binMeta.setColsInBlock(cols);
							binMeta.setRowsInBlock(cols);
							binOp.accept(this);
						}
					}
				}
			}
		}
		//TODO: move this outside to GlobalTransformationOptimizer
		this.applyChanges();
		return toModify;
	}

	public void resetVisitStatus(HopsDag toReset) {
		Map<String, Set<Hop>> hopsDir = toReset.getHopsDirectory();
		
		for(Set<Hop> val : hopsDir.values()) {
			if(val != null && !val.isEmpty()) {
				for(Hop h : val) {
					h.resetVisitStatus(this);
				}
			}
		}
	}

	@Override
	public Flag postVisit(Hop hops) {
		return Flag.GO_ON;
	}

	@Override
	public Flag visit(Hop hops) {
		return Flag.GO_ON;
	}

	@Override
	public boolean traverseBackwards() {
		return false;
	}

	@Override
	public void applyChanges() {
		if(LOG.isDebugEnabled())
		{
			LOG.debug("Applying changes for " + this.toString());
		}
		for(HopsMetaData meta : this.changes) {
			if(meta.isReblockRequired()) {
				@SuppressWarnings("unused")
				ReblockOp reblock = new ReblockOp(meta.getOperator(), (int)meta.getRowsInBlock(), (int)meta.getColsInBlock());
				LOG.info("Adding reblock for non matching block sizes...");
				continue;
			}
			
			Hop operator = meta.getOperator();
			operator.set_cols_in_block(meta.getColsInBlock());
			operator.set_rows_in_block(meta.getRowsInBlock());
			
			
		}
		
	}
}
