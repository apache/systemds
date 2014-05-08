/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.MemoTable;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.ReblockOp;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.FormatType;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.parser.Expression.DataType;

/**
 * This is an intermediate solution to reworking the rewrites - essentially 
 * the whole notation of a rewrite is redundant and can be replaced by direct use of
 * RewriteConfig plus logic in this plan rewriter
 * 
 */
public abstract class PlanRewriter 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static void applyToHop(Hop hop, RewriteConfigSet rcs)
	{
		for(RewriteConfig rc : rcs.getConfigs()) {
			applyToHop(hop, rc);
		}
	}
	
	public static void applyToHop(Hop hop, RewriteConfig rc)
	{
		switch( rc.getType() )
		{
			case BLOCK_SIZE:
				hop.set_cols_in_block(rc.getValue());
				hop.set_rows_in_block(rc.getValue());
				break;
			case EXEC_TYPE:
				ExecType eType = ExecType.MR;
				if( rc.getValue() == ExecType.CP.ordinal() )
					eType = ExecType.CP;
				if( rc.getValue()== ExecType.INVALID.ordinal() )
					eType = null;
				hop.setForcedExecType(eType);
				break;
			case FORMAT_CHANGE:
				//should only apply to FuncOps and here the statement block is needed
				//generate Reblocks for the rest???
				break;
		}
	}
	
	
	public static RewriteConfig extractRewriteConfigSetFromHop(Hop hop)
	{
		//TODO
		return null;
	}
	
	
	/**
	 * 
	 * @param hop
	 * @param type
	 * @return
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	public static RewriteConfig extractRewriteConfigFromHop(Hop hop, RewriteConfigType type)
	{
		RewriteConfig ret = null;
		
		switch( type )
		{
			case BLOCK_SIZE:
				ret = new RewriteConfigBlocksize((int)hop.get_rows_in_block());
				break;
			case EXEC_TYPE:
				ExecType extractedExecType = ExecType.INVALID; //unknown
				if(hop.getForcedExecType() != null)
					extractedExecType = hop.getForcedExecType();
				ret = new RewriteConfigExecType(extractedExecType.ordinal());
				break;
			case FORMAT_CHANGE:
				try{
					Lop lop = hop.constructLops();
					Format format = lop.getOutputParameters().getFormat();
					FormatType extractedFormat = FormatType.BINARY_BLOCK;
					if(format == Format.TEXT) {
						extractedFormat = FormatType.TEXT_CELL;
					} else if(hop.get_cols_in_block() == -1L) {
						extractedFormat = FormatType.BINARY_CELL;
					}
					ret = new RewriteConfigFormat(extractedFormat.ordinal());
				}
				catch(DMLException ex){
					ex.printStackTrace();
				}
				break;
		}
		
		
		return ret;
	}
	
	public static void generateRewrites(Hop hop, OptimizedPlan plan, RewriteConfigSet rcs) {
		RewriteConfig blockSize = rcs.getConfigByType(RewriteConfigType.BLOCK_SIZE);
		Rewrite reblockRewrite = plan.getRewrite(RewriteConfigType.BLOCK_SIZE);
		
		RewriteConfig locationParam = rcs.getConfigByType(RewriteConfigType.EXEC_TYPE);
		LocationRewrite locationRewrite = new LocationRewrite();
		locationRewrite.setExecLocation(locationParam.getValue());
		plan.addRewrite(RewriteConfigType.EXEC_TYPE, locationRewrite);
		
		RewriteConfig formatParam = rcs.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		
		//in this case no reblock is required but just setting the current block size
		//TODO both should be better separated
		if(reblockRewrite == null || !(reblockRewrite instanceof ReblockRewrite)) {
			BlockSizeRewrite rewrite = new BlockSizeRewrite();
			rewrite.setToBlockSize(blockSize.getValue());
			plan.addRewrite(RewriteConfigType.BLOCK_SIZE, rewrite);
			
			ReblockRewrite formatRewrite = new ReblockRewrite();
			formatRewrite.setFormat(FormatType.values()[formatParam.getValue()]);
			formatRewrite.setToBlockSize(blockSize.getValue());
			plan.addRewrite(RewriteConfigType.FORMAT_CHANGE, formatRewrite);
			
		}else {
			((ReblockRewrite)reblockRewrite).setFormat(FormatType.values()[formatParam.getValue()]);
		}
		
	}
	
	public static boolean isValidForOperator(Hop hop, RewriteConfigSet rcs) {
		for(RewriteConfig rc : rcs.getConfigs()) {
			if(!isValidForOperator(hop,rc))
				return false;
		}
		
		return true;
	}
	
	public static boolean isValidForOperator(Hop hop, RewriteConfig rc) 
	{
		if( rc.getType()!=RewriteConfigType.EXEC_TYPE )
			return true;
			
		if(rc.getValue()== ExecType.CP.ordinal()) {
			if(hop instanceof ReblockOp ) {
				return false;
			}
		
			if(hop instanceof DataOp) { 
					DataOp data = (DataOp)hop;
					if((data.get_dataop() == DataOpTypes.PERSISTENTREAD || data.get_dataop() == DataOpTypes.PERSISTENTWRITE)
							&& rc.getValue()==ExecType.CP.ordinal() 
							&& !data.get_dataType().equals(DataType.SCALAR)) {
						System.out.println("data op in CP: " + data + ", " + data.get_dataop());
						return false;
					}
					
					//read matrices always from HDFS
					if((data.get_dataop() == DataOpTypes.TRANSIENTREAD || data.get_dataop() == DataOpTypes.TRANSIENTWRITE) 
							&& data.get_dataType() == DataType.MATRIX 
							&& rc.getValue()==ExecType.CP.ordinal()) {
						return false;
					}
					
			}
			
			if(hop instanceof LiteralOp && rc.getValue() == ExecType.MR.ordinal()) {
				return false;
			}
			
			hop.refreshMemEstimates(new MemoTable());
			double memEstimate = hop.getMemEstimate();
			double budget = OptimizerUtils.getLocalMemBudget();
			if(memEstimate > budget) 
				return false;
		}
		
		return true;
	}
	
	public static boolean isFormatValid(Hop hop, RewriteConfigFormat rcf) 
	{
		if(rcf.getValue()==FormatType.BINARY_BLOCK.ordinal()) {
			return true;
		}
		
		if(hop instanceof CrossBlockOp) {
			return true;
		}
		
		if(hop instanceof FunctionOp) {
			if(!(rcf.getValue()==FormatType.BINARY_CELL.ordinal())) {
				return true;
			}
		}
		
		if(hop instanceof DataOp) {
			return true;
		}
		return false;
	}
}
