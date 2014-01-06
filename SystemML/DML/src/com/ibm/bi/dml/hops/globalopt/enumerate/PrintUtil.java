/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.Data;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.OutputParameters;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.parser.Expression.DataType;

public class PrintUtil 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final String TAB_CONSTANT = "   ";

	public static String generateLopString(Lop lops) {
		StringBuilder builder = new StringBuilder();
		
		if(lops instanceof ReBlock)
		{
			builder.append(lops);
		}
		builder.append(lops);
		if(lops instanceof Data) {
			HashMap<String, Lop> inputParams = ((Data)lops).getInputParams();
			if(inputParams != null) {
				for(Entry<String, Lop> e : inputParams.entrySet()) {
					builder.append(e.getKey() + " - " + generateLopString(e.getValue())); 
				}
			}
		}

		OutputParameters outputParams = lops.getOutputParameters();
		
		if(outputParams != null) {
			builder.append(outputParams.get_rows_in_block() + " x " + outputParams.get_cols_in_block());
		}
		return builder.toString();
	}
	
	public static String generateLopDagString(Lop lops) {
		StringBuilder builder = new StringBuilder();
		if(lops != null) {
			ArrayList<Lop> inputs = lops.getInputs();
			if(inputs != null) {
				for(Lop l : inputs) {
					if(l instanceof Data) {
						Data data = (Data)l;
						if(!data.get_dataType().equals(DataType.MATRIX))
						{	
							continue;
						}
					}
					builder. append("(" + generateLopDagString(l) + "), ");
				}
				//TODO: generate lop connections similar to CBH on Hops level
			}
			
			OutputParameters outputParams = lops.getOutputParameters();
			builder.append(lops.getID() + " " + lops.getClass().getSimpleName() + " " + outputParams.getFormat() + " " + outputParams.get_rows_in_block() + " x " + outputParams.get_cols_in_block());
		}
		return builder.toString();
	}
	
	public static String printMemoEntry(MemoEntry entry, int lvl) {
		StringBuilder builder = new StringBuilder();
		OptimizedPlan plan = entry.getOptPlan();
		
		for(int i = 0; i < lvl; i++) {
			builder.append(TAB_CONSTANT);
		}
		builder.append("lopID: " + entry.getLopId() + "\n");
		
		Hop operator = plan.getOperator();
		Lop generatedLop = plan.getGeneratedLop();
		if(generatedLop != null) {
			for(int i = 0; i < lvl; i++) {
				builder.append(TAB_CONSTANT);
			}
			if(operator != null) {
				builder.append("" + operator.get_name() + " ");
			}
			builder.append("lop: " + PrintUtil.generateLopDagString(generatedLop) + "\n");
		}else {
			
			if(operator != null) {
				for(int i = 0; i < lvl; i++) {
					builder.append(TAB_CONSTANT);
				}
				builder.append("hop: " + operator.getClass().getSimpleName() + ", " + operator.get_name() + "\n");
			}
		}
		
		RewriteConfigSet config = plan.getConfig();
		if(config != null) {
			for(int i = 0; i < lvl; i++) {
				builder.append(TAB_CONSTANT);
			}
			builder.append("config: " + config + "\n");
		}
		
		for(int i = 0; i < lvl; i++) {
			builder.append("  ");
		}
		builder.append("cost: " +  plan.getCost() + "\n");
		for(int i = 0; i < lvl; i++) {
			builder.append(TAB_CONSTANT);
		}
		builder.append("cumulated cost: " +  plan.getCumulatedCost() + "\n");
		
		List<MemoEntry> inputPlans = plan.getInputPlans();
		if(inputPlans != null && inputPlans.size() > 0) {
			for(int i = 0; i < lvl; i++) {
				builder.append("\n  ");
			}
			builder.append("[\n");
			
			for(MemoEntry e : inputPlans) {
				for(int i = 0; i < lvl; i++) {
					builder.append(TAB_CONSTANT);
				}
				builder.append(PrintUtil.printMemoEntry(e, ++lvl));
				--lvl;
				builder.append("\n");
			}
			
			for(int i = 0; i < lvl; i++) {
				builder.append("  ");
			}
			builder.append("\n]\n");
		}
		return builder.toString();
	}
	
	public static String printLopParameters(Hop hop) {
		StringBuilder builder = new StringBuilder();
		
		builder.append(hop.get_name() );
		
		Lop lop = hop.get_lops();
		if(lop == null) {
			builder.append("NULL Lop");
		} else {
			builder.append("input format" + ((Data)lop).getInputParams());
			builder.append("output format" + lop.getOutputParameters().getFormat());
		}
		
		return builder.toString();
	}
	
	public static String printCostMap(Map<Double, OptimizedPlan> costMap) {
		StringBuilder builder = new StringBuilder();
		if(costMap == null) {
			return builder.toString();
		}
		
		builder.append("size: " + costMap.size() + "\n");
		for(Entry<Double, OptimizedPlan> entry : costMap.entrySet()) {
			OptimizedPlan value = entry.getValue();
			String lopString = PrintUtil.generateLopDagString(value.getGeneratedLop());
			RewriteConfig blockSizeParam = value.getConfig().getConfigByType(RewriteConfigType.BLOCK_SIZE);
			
			builder.append(entry.getKey() + " " + blockSizeParam.getValue() + " " + lopString);
			builder.append("\n");
		}
		return builder.toString();
	}
}
