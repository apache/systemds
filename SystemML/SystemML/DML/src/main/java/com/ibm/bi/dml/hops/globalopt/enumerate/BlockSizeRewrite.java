/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.OutputParameters;

/**
 * Just set block size without adding a reblock.
 */
public class BlockSizeRewrite extends Rewrite 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(BlockSizeRewrite.class);
	private long toBlockSize = -1L;
	
	/* (non-Javadoc)
	 * @see com.ibm.bi.dml.optimizer.enumeration.Rewrite#apply(com.ibm.bi.dml.optimizer.enumeration.OptimizedPlan)
	 */
	@Override
	public void apply(OptimizedPlan plan) {
		Hop operator = plan.getOperator();
		if((operator instanceof DataOp) 
				&& ((DataOp)operator).get_dataop().equals(DataOpTypes.PERSISTENTWRITE)) {
			return;
		}
		
		operator.setColsInBlock((long)this.toBlockSize);
		operator.setRowsInBlock((long)this.toBlockSize);
		operator.setLops(null);
		Lop constructedLop;
		
		try {
			constructedLop = operator.constructLops();
			if(operator instanceof CrossBlockOp) {
				((CrossBlockOp)operator).addRewrite(this);
			}
			
			OutputParameters outputParameters = constructedLop.getOutputParameters();
			Long nnz = outputParameters.getNnz();
			Long numCols = outputParameters.getNumCols();
			Long numRows = outputParameters.getNumRows();
			outputParameters.setDimensions(numRows, numCols, this.toBlockSize, this.toBlockSize, nnz);
			
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		} catch (LopsException e) {
			LOG.error(e.getMessage(), e);
		}
		
	}

	public long getToBlockSize() {
		return toBlockSize;
	}

	public void setToBlockSize(long toBlockSize) {
		this.toBlockSize = toBlockSize;
	}

}
