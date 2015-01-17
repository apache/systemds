/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.hops.globalopt.enumerate;

import java.util.HashSet;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.globalopt.CrossBlockOp;
import com.ibm.bi.dml.hops.globalopt.enumerate.InterestingProperty.FormatType;
import com.ibm.bi.dml.hops.globalopt.enumerate.RewriteConfig.RewriteConfigType;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.OutputParameters;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.lops.OutputParameters.Format;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.controlprogram.ExternalFunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;

public class ReblockRewrite extends Rewrite 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	
	private static final Log LOG = LogFactory.getLog(ReblockRewrite.class);

	private int fromBlockSize = -1;
	private int toBlockSize = -1;

	private int value;

	@Override
	public void apply(OptimizedPlan plan) {
		try {

			Hop operator = plan.getOperator();

			if (operator instanceof FunctionOp) {
				handleTextInputsForExternalFunctions(plan, operator);
				return;
			}
			if ((operator instanceof DataOp)
					&& ((DataOp) operator).get_dataop().equals(
							DataOpTypes.PERSISTENTWRITE)) {
				return;
			}
			Lop constructedLop = operator.constructLops();
			apply(operator, constructedLop, plan);
			plan.setGeneratedLop(operator.getLops());
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		} catch (LopsException e) {
			LOG.error(e.getMessage(), e);
		}
	}

	/**
	 * TODO: this logic is way to complex due to special cases
	 * a general, more elegant solution is needed.
	 * 
	 * @param plan
	 * @param operator
	 * @param constructedLop
	 * @throws HopsException
	 * @throws LopsException
	 */
	public void apply(Hop operator, Lop constructedLop, OptimizedPlan plan) {
		//FIXME: ugly but necessary, however, should be fixed in the logic of creating rewrites
		if(operator instanceof DataOp 
				&& (((DataOp)operator).get_dataop().equals(DataOpTypes.TRANSIENTWRITE)
				|| (((DataOp)operator).get_dataop().equals(DataOpTypes.PERSISTENTWRITE)))
				){
			
			handleTransientWrites(operator, constructedLop, plan);
			return;
		}
		
		if(operator instanceof CrossBlockOp) {
			CrossBlockOp cross = (CrossBlockOp)operator;
			cross.addRewrite(this);
			return;
		}
		
		if(operator instanceof DataOp 
				&& ((DataOp)operator).get_dataop().equals(DataOpTypes.TRANSIENTREAD)) {
			
			handleTransientReads(operator, constructedLop, plan);
			return;
		}
		
		if (value == FormatType.BINARY_BLOCK.ordinal()) {
			
			if(this.fromBlockSize != -1L) {
				operator.setColsInBlock(this.fromBlockSize);
				operator.setRowsInBlock(this.fromBlockSize);
				operator.setLops(null);
				
				try {
					constructedLop = operator.constructLops();
				} catch (HopsException e) {
					LOG.error(e.getMessage(), e);
				} catch (LopsException e) {
					LOG.error(e.getMessage(), e);
				}
				
			}
			
			
			OutputParameters outputParameters = constructedLop.getOutputParameters();
			Long lopBlockSize = outputParameters.getColsInBlock();
			long hopBlockSize = operator.getColsInBlock();
			if(lopBlockSize == hopBlockSize && hopBlockSize == this.toBlockSize) {
				return;
			}
			
			
			if (this.toBlockSize == -1L) {
				return;
			}

			if (constructedLop instanceof ReBlock) {
				ReBlock reblock = (ReBlock) constructedLop;
				reblock.getOutputParameters().setFormat(Format.BINARY);
				long rows = reblock.getOutputParameters().getNumRows();
				long cols = reblock.getOutputParameters().getNumCols();
				Long nnz = reblock.getOutputParameters().getNnz();
				try {
					reblock.getOutputParameters().setDimensions(rows, cols,
							this.toBlockSize, this.toBlockSize, nnz);
				} catch (HopsException e) {
					LOG.error(e.getMessage(), e);
				}
			} else if (constructedLop instanceof DataGen) {
				DataGen dataGen = (DataGen) constructedLop;
				dataGen.getOutputParameters().setFormat(Format.BINARY);
				long rows = dataGen.getOutputParameters().getNumRows();
				long cols = dataGen.getOutputParameters().getNumCols();
				Long nnz = dataGen.getOutputParameters().getNnz();
				try {
					dataGen.getOutputParameters().setDimensions(rows, cols,
							this.toBlockSize, this.toBlockSize, nnz);
				} catch (HopsException e) {
					LOG.error(e.getMessage(), e);
				}
			} else {

				ReBlock reblock;
				try {
					reblock = new ReBlock(constructedLop, (long)this.toBlockSize,
							(long)this.toBlockSize, operator.getDataType(), operator
									.getValueType(), true);
					reblock.getOutputParameters().setFormat(Format.BINARY);
					reblock.getOutputParameters().setDimensions(
							operator.getDim1(), operator.getDim2(),
							this.toBlockSize, this.toBlockSize,
							operator.getNnz());
					reblock.setAllPositions(operator.getBeginLine(), operator
							.getBeginColumn(), operator.getEndLine(), operator
							.getEndColumn());
					operator.setLops(reblock);
				} catch (LopsException e) {
					LOG.error(e.getMessage(), e);
				} catch (HopsException e) {
					LOG.error(e.getMessage(), e);
				}
			}
			return;
		} else {
			// BINARY CELL/TEXT

			if ((operator instanceof DataOp)
					&& !((DataOp) operator).get_dataop().equals(
							DataOpTypes.PERSISTENTWRITE)
					&& !((DataOp) operator).get_dataop().equals(
							DataOpTypes.TRANSIENTWRITE)) {
				DataOp data = (DataOp)operator;
				// if the current operator cannot process text/BinCell then it
				// takes Binblock from the
				// input and appends a reblock
				
				if(value==FormatType.TEXT_CELL.ordinal() && data.getFormatType().equals(FileFormatTypes.TEXT)) {
					return;
				}
				
				if(value==FormatType.BINARY_CELL.ordinal() && data.getFormatType().equals(FileFormatTypes.TEXT)) {
					return;
				}
				if(value==FormatType.TEXT_CELL.ordinal() 
						&& data.getFormatType().equals(FileFormatTypes.BINARY) 
						&& data.getColsInBlock() == -1L
				) {
					return;
				}
				
				if (true) {//(!this.format.isFormatValid(operator)) {
					Hop input = operator.getInput().get(0);
					Lop inputLop = input.getLops();
					if (inputLop.getOutputParameters().getColsInBlock() > -1L) {
						operator.setColsInBlock(input.getColsInBlock());
						operator.setRowsInBlock(input.getRowsInBlock());
						operator.setLops(null);
						try {
							constructedLop = operator.constructLops();
						} catch (HopsException e) {
							LOG.error(e.getMessage(), e);
						} catch (LopsException e) {
							LOG.error(e.getMessage(), e);
						}
					}
				}

				ReBlock reblock;
				try {
					reblock = new ReBlock(constructedLop, -1L, -1L, operator
							.getDataType(), operator.getValueType(), true);
					if (value==FormatType.TEXT_CELL.ordinal()) {
						reblock.getOutputParameters().setFormat(Format.TEXT);
					} else {
						reblock.getOutputParameters().setFormat(Format.BINARY);
					}

					reblock.getOutputParameters().setDimensions(
							operator.getDim1(), operator.getDim2(), -1, -1,
							operator.getNnz());
					reblock.setAllPositions(operator.getBeginLine(), operator
							.getBeginColumn(), operator.getEndLine(), operator
							.getEndColumn());
					operator.setLops(reblock);
				} catch (LopsException e) {
					LOG.error(e.getMessage(), e);
				} catch (HopsException e) {
					LOG.error(e.getMessage(), e);
				}
			} else {
				
//				if(plan.getInputPlans().size() > 0) {
//					List<MemoEntry> inputPlans = plan.getInputPlans();
//					MemoEntry firstInput = inputPlans.get(0);
//					Lops inputLop = firstInput.getRootLop();
//					long inputBlockSize = inputLop.getOutputParameters().getColsInBlock();
//					
//					if(inputBlockSize == this.fromBlockSize) {
//						System.out.println("redundant lop access for " + operator);
//					}
					
//					operator.setColsInBlock(inputBlockSize);
//					operator.setRowsInBlock(inputBlockSize);
					operator.setColsInBlock(this.fromBlockSize);
					operator.setRowsInBlock(this.fromBlockSize);
				
					operator.setLops(null);
					
					try {
						constructedLop = operator.constructLops();
					} catch (HopsException e) {
						LOG.error(e.getMessage(), e);
					} catch (LopsException e) {
						LOG.error(e.getMessage(), e);
					}
//				}
				
				//operator is not instance of DataOP
				if (value==FormatType.BINARY_CELL.ordinal()) {
					//every normal operator can produce binary cell
					constructedLop.getOutputParameters().setFormat(Format.BINARY);
				} else {
					//in case of TEXT a reblock_text has to be appended
					ReBlock reblock;
					try {
						reblock = new ReBlock(constructedLop, -1L, -1L, operator
								.getDataType(), operator.getValueType(), true);
						reblock.getOutputParameters().setFormat(Format.TEXT);
						reblock.getOutputParameters().setDimensions(
								operator.getDim1(), operator.getDim2(), -1, -1,
								operator.getNnz());
						reblock.setAllPositions(operator.getBeginLine(), operator
								.getBeginColumn(), operator.getEndLine(), operator
								.getEndColumn());
						operator.setLops(reblock);
					} catch (LopsException e) {
						LOG.error(e.getMessage(), e);
					} catch (HopsException e) {
						LOG.error(e.getMessage(), e);
					}
				}
			}
		}
	}

	/**
	 * @param operator
	 * @param constructedLop
	 * @param plan
	 * @return
	 */
	private Lop handleTransientReads(Hop operator, Lop constructedLop,
			OptimizedPlan plan) {
		MemoEntry memoEntry = plan.getInputPlans().get(0);
		RewriteConfigSet inputConfig = memoEntry.getConfig();
		RewriteConfig inputFormat = inputConfig.getConfigByType(RewriteConfigType.FORMAT_CHANGE);
		RewriteConfig inputBlockSize = inputConfig.getConfigByType(RewriteConfigType.BLOCK_SIZE);
		
		DataOp data = (DataOp)operator;
		if(inputFormat.getValue()==FormatType.TEXT_CELL.ordinal()) {
			data.setFormatType(FileFormatTypes.TEXT);
		}else {
			data.setFormatType(FileFormatTypes.BINARY);
		}
		data.setColsInBlock(inputBlockSize.getValue());
		data.setRowsInBlock(inputBlockSize.getValue());
		data.setLops(null);
		try {
			constructedLop = data.constructLops();
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		} catch (LopsException e) {
			LOG.error(e.getMessage(), e);
		}
			
		if(value==inputFormat.getValue() && this.toBlockSize == inputBlockSize.getValue()) {
			plan.setGeneratedLop(constructedLop);
		} else {
		
		ReBlock reblock;
		try {
			reblock = new ReBlock(constructedLop, (long)this.toBlockSize,
					(long)this.toBlockSize, operator.getDataType(), operator
							.getValueType(), true);
			reblock.getOutputParameters().setFormat(Format.BINARY);
			reblock.getOutputParameters().setDimensions(
					operator.getDim1(), operator.getDim2(),
					this.toBlockSize, this.toBlockSize,
					operator.getNnz());
			reblock.setAllPositions(operator.getBeginLine(), operator
					.getBeginColumn(), operator.getEndLine(), operator
					.getEndColumn());
			operator.setLops(reblock);
			plan.setGeneratedLop(reblock);
		} catch (LopsException e) {
			LOG.error(e.getMessage(), e);
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		}
		}
		return constructedLop;
	}

	/**
	 * @param operator
	 * @param constructedLop
	 * @param plan
	 * @return
	 */
	private Lop handleTransientWrites(Hop operator, Lop constructedLop,
			OptimizedPlan plan) {
		DataOp data = (DataOp)operator;
		data.setColsInBlock(this.toBlockSize);
		data.setRowsInBlock(this.toBlockSize);
		if(value==FormatType.TEXT_CELL.ordinal()) {
			data.setFormatType(FileFormatTypes.TEXT);
		}else {
			data.setFormatType(FileFormatTypes.BINARY);
		}
		try {
			data.setLops(null);
			constructedLop = data.constructLops();
			plan.setGeneratedLop(constructedLop);
		} catch (HopsException e) {
			LOG.error(e.getMessage(), e);
		} catch (LopsException e) {
			LOG.error(e.getMessage(), e);
		}
		return constructedLop;
	}

	/**
	 * Extract the {@link ExternalFunctionProgramBlock} and set skipReblock for
	 * every input that has TEXT as format.
	 * 
	 * @param plan
	 * @param operator
	 */
	private void handleTextInputsForExternalFunctions(OptimizedPlan plan,
			Hop operator) {
		FunctionOp externalFunction = (FunctionOp) operator;
		String functionName = externalFunction.getFunctionName();
		String nameSpace = externalFunction.getFunctionNamespace();
		Program dmlProgram = plan.getRuntimeProgram();
		try {
			FunctionProgramBlock functionProgramBlock = dmlProgram
					.getFunctionProgramBlock(nameSpace, functionName);
			if (functionProgramBlock instanceof ExternalFunctionProgramBlock) {
				ExternalFunctionProgramBlock externalBlock = (ExternalFunctionProgramBlock) functionProgramBlock;

				Set<String> varsIn = new HashSet<String>();
				Set<String> varsOut = new HashSet<String>();

				for (MemoEntry inputPlan : plan.getInputPlans()) {
					RewriteConfig format = inputPlan.getConfig().getConfigByType(RewriteConfigType.FORMAT_CHANGE);
					Hop inputHop = inputPlan.getRootHop();
					if (format.getValue()==FormatType.TEXT_CELL.ordinal()) {
						varsIn.add(inputHop.getName());
					}
				}
				
				externalBlock.setSkippedReblockLists(varsIn, varsOut);
			}
		} catch (DMLRuntimeException e) {
			LOG.error(e.getMessage(), e);
		}
	}

	public long getFromBlockSize() {
		return fromBlockSize;
	}

	public void setFromBlockSize(int fromBlockSize) {
		this.fromBlockSize = fromBlockSize;
	}

	public long getToBlockSize() {
		return toBlockSize;
	}

	public void setToBlockSize(int toBlockSize) {
		this.toBlockSize = toBlockSize;
	}

	public FormatType getFormat() {
		return FormatType.values()[value];
	}

	public void setFormat(FormatType format) {
		value = format.ordinal();
	}

}
