/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package org.apache.sysml.hops;

import java.util.HashMap;
import java.util.Map.Entry;

import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.lops.Aggregate;
import org.apache.sysml.lops.Data;
import org.apache.sysml.lops.Group;
import org.apache.sysml.lops.GroupedAggregate;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.lops.OutputParameters.Format;
import org.apache.sysml.lops.ParameterizedBuiltin;
import org.apache.sysml.lops.RepMat;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.util.UtilFunctions;


/**
 * Defines the HOP for calling an internal function (with custom parameters) from a DML script. 
 * 
 */
public class ParameterizedBuiltinOp extends Hop 
{
	
	private static boolean COMPILE_PARALLEL_REMOVEEMPTY = true;
	public static boolean FORCE_DIST_RM_EMPTY = false;

	//operator type
	private ParamBuiltinOp _op;
	
	//removeEmpty hints
	private boolean _outputEmptyBlocks = true;
	private boolean _outputPermutationMatrix = false;

	private boolean _bRmEmptyBC = false;
	
	/**
	 * List of "named" input parameters. They are maintained as a hashmap:
	 * parameter names (String) are mapped as indices (Integer) into getInput()
	 * arraylist.
	 * 
	 * i.e., getInput().get(_paramIndexMap.get(parameterName)) refers to the Hop
	 * that is associated with parameterName.
	 */
	private HashMap<String, Integer> _paramIndexMap = new HashMap<String, Integer>();

	private ParameterizedBuiltinOp() {
		//default constructor for clone
	}

	/**
	 * Creates a new HOP for a function call
	 */
	public ParameterizedBuiltinOp(String l, DataType dt, ValueType vt,
			ParamBuiltinOp op, HashMap<String, Hop> inputParameters) {
		super(l, dt, vt);

		_op = op;

		int index = 0;
		for( Entry<String,Hop> e : inputParameters.entrySet() ) 
		{
			String s = e.getKey();
			Hop input = e.getValue();
			getInput().add(input);
			input.getParent().add(this);

			_paramIndexMap.put(s, index);
			index++;
		}
		
		//compute unknown dims and nnz
		refreshSizeInformation();
	}

	public HashMap<String, Integer> getParamIndexMap(){
		return _paramIndexMap;
	}
		
	@Override
	public String getOpString() {
		return "" + _op;
	}
	
	public ParamBuiltinOp getOp() {
		return _op;
	}

	public void setOutputEmptyBlocks(boolean flag)
	{
		_outputEmptyBlocks = flag;
	}
	
	public void setOutputPermutationMatrix(boolean flag)
	{
		_outputPermutationMatrix = flag;
	}
	
	public Hop getTargetHop()
	{
		Hop targetHop = getInput().get(_paramIndexMap.get("target"));
		
		return targetHop;
	}
	
	@Override
	public Lop constructLops() 
		throws HopsException, LopsException 
	{		
		//return already created lops
		if( getLops() != null )
			return getLops();
		
		// construct lops for all input parameters
		HashMap<String, Lop> inputlops = new HashMap<String, Lop>();
		for (Entry<String, Integer> cur : _paramIndexMap.entrySet()) {
			inputlops.put(cur.getKey(), getInput().get(cur.getValue())
					.constructLops());
		}

		if ( _op == ParamBuiltinOp.CDF || _op == ParamBuiltinOp.INVCDF ) 
		{
			// simply pass the hashmap of parameters to the lop
			// set the lop for the function call (always CP)
			
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,
					HopsParameterizedBuiltinLops.get(_op), getDataType(),
					getValueType());
			
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
			setLops(pbilop);
		} 
		else if (_op == ParamBuiltinOp.GROUPEDAGG) 
		{
			ExecType et = optFindExecType();
			
			constructLopsGroupedAggregate(inputlops, et);
		}
		else if( _op == ParamBuiltinOp.RMEMPTY ) 
		{
			ExecType et = optFindExecType();
			et = (et == ExecType.MR && !COMPILE_PARALLEL_REMOVEEMPTY ) ? ExecType.CP_FILE : et;
			
			constructLopsRemoveEmpty(inputlops, et);
		} 
		else if(   _op == ParamBuiltinOp.REPLACE ) 
		{
			ExecType et = optFindExecType();
			
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);
			
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
			setLops(pbilop);
		}
		else if( _op == ParamBuiltinOp.REXPAND ) 
		{
			ExecType et = optFindExecType();
			
			constructLopsRExpand(inputlops, et);
		} 
		else if ( _op == ParamBuiltinOp.TRANSFORM ) 
		{
			ExecType et = optFindExecType();
			
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
			// output of transform is always in CSV format
			// to produce a blocked output, this lop must be 
			// fed into CSV Reblock lop.
			pbilop.getOutputParameters().setFormat(Format.CSV);
			setLops(pbilop);
		}

		//add reblock/checkpoint lops if necessary
		constructAndSetLopsDataFlowProperties();
				
		return getLops();
	}
	
	private void constructLopsGroupedAggregate(HashMap<String, Lop> inputlops, ExecType et) 
		throws HopsException, LopsException 
	{
		//reset reblock requirement (see MR aggregate / construct lops)
		setRequiresReblock( false );
		
		if ( et == ExecType.MR ) 
		{
			// construct necessary lops: combineBinary/combineTertiary and
			// groupedAgg

			boolean isWeighted = (_paramIndexMap.get(Statement.GAGG_WEIGHTS) != null);
			if (isWeighted) 
			{
				Lop append = BinaryOp.constructAppendLopChain(
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_GROUPS)),
						getInput().get(_paramIndexMap.get(Statement.GAGG_WEIGHTS)),
						DataType.MATRIX, getValueType(), true,
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)));

				// add the combine lop to parameter list, with a new name "combinedinput"
				inputlops.put(GroupedAggregate.COMBINEDINPUT, append);
				inputlops.remove(Statement.GAGG_TARGET);
				inputlops.remove(Statement.GAGG_GROUPS);
				inputlops.remove(Statement.GAGG_WEIGHTS);

			} 
			else 
			{
				Lop append = BinaryOp.constructMRAppendLop(
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)), 
						getInput().get(_paramIndexMap.get(Statement.GAGG_GROUPS)), 
						DataType.MATRIX, getValueType(), true,
						getInput().get(_paramIndexMap.get(Statement.GAGG_TARGET)));
				
				// add the combine lop to parameter list, with a new name
				// "combinedinput"
				inputlops.put(GroupedAggregate.COMBINEDINPUT, append);
				inputlops.remove(Statement.GAGG_TARGET);
				inputlops.remove(Statement.GAGG_GROUPS);

			}
			
			int colwise = -1;
			long outputDim1=-1, outputDim2=-1;
			Lop numGroups = inputlops.get(Statement.GAGG_NUM_GROUPS);
			if ( !dimsKnown() && numGroups != null && numGroups instanceof Data && ((Data)numGroups).isLiteral() ) {
				long ngroups = ((Data)numGroups).getLongValue();
				
				Lop input = inputlops.get(GroupedAggregate.COMBINEDINPUT);
				long inDim1 = input.getOutputParameters().getNumRows();
				long inDim2 = input.getOutputParameters().getNumCols();
				if(inDim1 > 0 && inDim2 > 0 ) {
					if ( inDim1 > inDim2 )
						colwise = 1;
					else 
						colwise = 0;
				}
				
				if ( colwise == 1 ) {
					outputDim1 = ngroups;
					outputDim2 = 1;
				}
				else if ( colwise == 0 ) {
					outputDim1 = 1;
					outputDim2 = ngroups;
				}
				
			}
			
			GroupedAggregate grp_agg = new GroupedAggregate(inputlops, getDataType(), getValueType());
			
			// output dimensions are unknown at compilation time
			grp_agg.getOutputParameters().setDimensions(outputDim1, outputDim2, getRowsInBlock(), getColsInBlock(), -1);
			grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());

			setLops(grp_agg);
			setRequiresReblock( true );
		}
		else //CP 
		{
			GroupedAggregate grp_agg = new GroupedAggregate(inputlops,
					getDataType(), getValueType(), et);
			// output dimensions are unknown at compilation time
			grp_agg.getOutputParameters().setDimensions(-1, -1, -1, -1, -1);
			grp_agg.setAllPositions(this.getBeginLine(), this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
			
			// introduce a reblock lop only if it is NOT single_node execution
			if( et == ExecType.CP){
				//force blocked output in CP (see below)
				grp_agg.getOutputParameters().setDimensions(-1, 1, getRowsInBlock(), getColsInBlock(), -1);
			}
			else if(et == ExecType.SPARK) {
				grp_agg.getOutputParameters().setDimensions(-1, 1, -1, -1, -1);
				setRequiresReblock( true );
			}
			//grouped agg, w/o reblock in CP
			setLops(grp_agg);
			
		}
	}

	/**
	 * 
	 * @param inputlops
	 * @param et
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsRemoveEmpty(HashMap<String, Lop> inputlops, ExecType et) 
		throws HopsException, LopsException 
	{
		Hop targetHop = getInput().get(_paramIndexMap.get("target"));
		Hop marginHop = getInput().get(_paramIndexMap.get("margin"));		
		Hop selectHop = (_paramIndexMap.get("select") != null) ? getInput().get(_paramIndexMap.get("select")):null;
		
		if( et == ExecType.CP || et == ExecType.CP_FILE )
		{
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops,HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
			setLops(pbilop);
			
			/*DISABLED CP PMM (see for example, MDA Bivar test, requires size propagation on recompile)
			if( et == ExecType.CP && isTargetDiagInput() && marginHop instanceof LiteralOp 
					 && ((LiteralOp)marginHop).getStringValue().equals("rows")
					 && _outputPermutationMatrix ) //SPECIAL CASE SELECTION VECTOR
			{				
				//TODO this special case could be taken into account for memory estimates in order
				// to reduce the estimates for the input diag and subsequent matrix multiply
				
				//get input vector (without materializing diag())
				Hop input = targetHop.getInput().get(0);
				long brlen = input.getRowsInBlock();
				long bclen = input.getColsInBlock();
				MemoTable memo = new MemoTable();
			
				boolean isPPredInput = (input instanceof BinaryOp && ((BinaryOp)input).isPPredOperation());
				
				//step1: compute index vectors
				Hop ppred0 = input;
				if( !isPPredInput ) { //ppred only if required
					ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp("0",0));
					HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
					ppred0.refreshSizeInformation();
					ppred0.computeMemEstimate(memo); //select exec type
					HopRewriteUtils.copyLineNumbers(this, ppred0);
				}
				
				UnaryOp cumsum = new UnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, ppred0); 
				HopRewriteUtils.setOutputBlocksizes(cumsum, brlen, bclen);
				cumsum.refreshSizeInformation(); 
				cumsum.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, cumsum);	
			
				BinaryOp sel = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, ppred0, cumsum);
				HopRewriteUtils.setOutputBlocksizes(sel, brlen, bclen); 
				sel.refreshSizeInformation();
				sel.computeMemEstimate(memo); //select exec type
				HopRewriteUtils.copyLineNumbers(this, sel);
				Lop loutput = sel.constructLops();
				
				//Step 4: cleanup hops (allow for garbage collection)
				HopRewriteUtils.removeChildReference(ppred0, input);
				
				setLops( loutput );
			}
			else //GENERAL CASE
			{
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin( et, inputlops,
						HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType());
				
				pbilop.getOutputParameters().setDimensions(getDim1(),getDim2(), getRowsInBlock(), getColsInBlock(), getNnz());
				setLineNumbers(pbilop);
				setLops(pbilop);
			}
			*/
		}
		else if( et == ExecType.MR )
		{
			//special compile for mr removeEmpty-diag 
			if(    isTargetDiagInput() && marginHop instanceof LiteralOp 
				&& ((LiteralOp)marginHop).getStringValue().equals("rows") )
	 		{
				//get input vector (without materializing diag())
				Hop input = targetHop.getInput().get(0);
				long brlen = input.getRowsInBlock();
				long bclen = input.getColsInBlock();
				MemoTable memo = new MemoTable();
			
				boolean isPPredInput = (input instanceof BinaryOp && ((BinaryOp)input).isPPredOperation());
				
				//step1: compute index vectors
				Hop ppred0 = input;
				if( !isPPredInput ) { //ppred only if required
					ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp(0));
					HopRewriteUtils.updateHopCharacteristics(ppred0, brlen, bclen, memo, this);
				}
				
				UnaryOp cumsum = new UnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, ppred0); 
				HopRewriteUtils.updateHopCharacteristics(cumsum, brlen, bclen, memo, this);
			
				Lop loutput = null;
				double mest = AggBinaryOp.getMapmmMemEstimate(input.getDim1(), 1, brlen, bclen, -1, brlen, bclen, brlen, bclen, -1, 1, true);
				double mbudget = OptimizerUtils.getRemoteMemBudgetMap(true);
				if( _outputPermutationMatrix && mest < mbudget ) //SPECIAL CASE: SELECTION VECTOR
				{
					BinaryOp sel = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, ppred0, cumsum);
					HopRewriteUtils.updateHopCharacteristics(sel, brlen, bclen, memo, this);
					loutput = sel.constructLops();
				}
				else //GENERAL CASE: GENERAL PERMUTATION MATRIX
				{
					//max ensures non-zero entries and at least one output row
					BinaryOp max = new BinaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MAX, cumsum, new LiteralOp(1));
					HopRewriteUtils.updateHopCharacteristics(max, brlen, bclen, memo, this);
					
					DataGenOp seq = HopRewriteUtils.createSeqDataGenOp(input);
					seq.setName("tmp4");
					HopRewriteUtils.updateHopCharacteristics(seq, brlen, bclen, memo, this);
					
					//step 2: compute removeEmpty(rows) output via table, seq guarantees right column dimension
					//note: weights always the input (even if isPPredInput) because input also includes 0s
					TernaryOp table = new TernaryOp("tmp5", DataType.MATRIX, ValueType.DOUBLE, OpOp3.CTABLE, max, seq, input);
					HopRewriteUtils.setOutputBlocksizes(table, brlen, bclen);
					table.refreshSizeInformation();
					table.setForcedExecType(ExecType.MR); //force MR 
					HopRewriteUtils.copyLineNumbers(this, table);
					table.setDisjointInputs(true);
					table.setOutputEmptyBlocks(_outputEmptyBlocks);
					loutput = table.constructLops();
					
					HopRewriteUtils.removeChildReference(table, input);	
				}
				
				//Step 4: cleanup hops (allow for garbage collection)
				HopRewriteUtils.removeChildReference(ppred0, input);
				
				setLops( loutput );
			}
			//default mr remove empty
			else if( et == ExecType.MR )
			{
				//TODO additional physical operator if offsets fit in memory
				
				if( !(marginHop instanceof LiteralOp) )
					throw new HopsException("Parameter 'margin' must be a literal argument.");
					
				Hop input = targetHop;
				long rlen = input.getDim1();
				long clen = input.getDim2();
				long brlen = input.getRowsInBlock();
				long bclen = input.getColsInBlock();
				long nnz = input.getNnz();
				boolean rmRows = ((LiteralOp)marginHop).getStringValue().equals("rows");
				
				//construct lops via new partial hop dag and subsequent lops construction 
				//in order to reuse of operator selection decisions
				
				BinaryOp ppred0 = null;
				Hop emptyInd = null;
				
				if(selectHop == null) {
					//Step1: compute row/col non-empty indicators 
					ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp(0));
					HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
					ppred0.refreshSizeInformation();
					ppred0.setForcedExecType(ExecType.MR); //always MR 
					HopRewriteUtils.copyLineNumbers(this, ppred0);
					
					emptyInd = ppred0;
					if( !((rmRows && clen == 1) || (!rmRows && rlen==1)) ){
						emptyInd = new AggUnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, AggOp.MAX, rmRows?Direction.Row:Direction.Col, ppred0);
						HopRewriteUtils.setOutputBlocksizes(emptyInd, brlen, bclen);
						emptyInd.refreshSizeInformation();
						emptyInd.setForcedExecType(ExecType.MR); //always MR
						HopRewriteUtils.copyLineNumbers(this, emptyInd);
					}
				} else {
					emptyInd = selectHop;
					HopRewriteUtils.setOutputBlocksizes(emptyInd, brlen, bclen);
					emptyInd.refreshSizeInformation();
					emptyInd.setForcedExecType(ExecType.MR); //always MR
					HopRewriteUtils.copyLineNumbers(this, emptyInd);
				}
				
				//Step 2: compute row offsets for non-empty rows
				Hop cumsumInput = emptyInd;
				if( !rmRows ){
					cumsumInput = HopRewriteUtils.createTranspose(emptyInd);
					HopRewriteUtils.updateHopCharacteristics(cumsumInput, brlen, bclen, this);	
				}
			
				UnaryOp cumsum = new UnaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, cumsumInput); 
				HopRewriteUtils.updateHopCharacteristics(cumsum, brlen, bclen, this);
			
				Hop cumsumOutput = cumsum;
				if( !rmRows ){
					cumsumOutput = HopRewriteUtils.createTranspose(cumsum);
					HopRewriteUtils.updateHopCharacteristics(cumsumOutput, brlen, bclen, this);	
				}
				
				Hop maxDim = new AggUnaryOp("tmp4", DataType.SCALAR, ValueType.DOUBLE, AggOp.MAX, Direction.RowCol, cumsumOutput); //alternative: right indexing
				HopRewriteUtils.updateHopCharacteristics(maxDim, brlen, bclen, this);
				
				BinaryOp offsets = new BinaryOp("tmp5", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, cumsumOutput, emptyInd);
				HopRewriteUtils.updateHopCharacteristics(offsets, brlen, bclen, this);
				
				//Step 3: gather non-empty rows/cols into final results 
				Lop linput = input.constructLops();
				Lop loffset = offsets.constructLops();
				Lop lmaxdim = maxDim.constructLops();
				
				boolean requiresRep =   ((clen>bclen || clen<=0) &&  rmRows) 
						             || ((rlen>brlen || rlen<=0) && !rmRows);
				
				if( requiresRep ) {
					Lop pos = createOffsetLop(input, rmRows); //ncol of left input (determines num replicates)
					loffset = new RepMat(loffset, pos, rmRows, DataType.MATRIX, ValueType.DOUBLE);
					loffset.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
					setLineNumbers(loffset);
				}
				
				Group group1 = new Group(linput, Group.OperationTypes.Sort, getDataType(), getValueType());
				setLineNumbers(group1);
				group1.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
			
				Group group2 = new Group( loffset, Group.OperationTypes.Sort, getDataType(), getValueType());
				setLineNumbers(group2);
				group2.getOutputParameters().setDimensions(rlen, clen, brlen, bclen, nnz);
			
				HashMap<String, Lop> inMap = new HashMap<String, Lop>();
				inMap.put("target", group1);
				inMap.put("offset", group2);
				inMap.put("maxdim", lmaxdim);
				inMap.put("margin", inputlops.get("margin"));
				
				ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inMap, HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);			
				setOutputDimensions(pbilop);
				setLineNumbers(pbilop);
			
				Group group3 = new Group( pbilop, Group.OperationTypes.Sort, getDataType(), getValueType());
				setLineNumbers(group3);
				group3.getOutputParameters().setDimensions(-1, -1, brlen, bclen, -1);
				
				Aggregate finalagg = new Aggregate(group3, Aggregate.OperationTypes.Sum, DataType.MATRIX, getValueType(), ExecType.MR);
				setOutputDimensions(finalagg);
				setLineNumbers(finalagg);
				
				//Step 4: cleanup hops (allow for garbage collection)
				if(selectHop == null)
					HopRewriteUtils.removeChildReference(ppred0, input);
				
				setLops(finalagg);
			}	
		}
		else if( et == ExecType.SPARK )
		{
			if( !(marginHop instanceof LiteralOp) )
				throw new HopsException("Parameter 'margin' must be a literal argument.");
				
			Hop input = targetHop;
			long rlen = input.getDim1();
			long clen = input.getDim2();
			long brlen = input.getRowsInBlock();
			long bclen = input.getColsInBlock();
			boolean rmRows = ((LiteralOp)marginHop).getStringValue().equals("rows");
			
			//construct lops via new partial hop dag and subsequent lops construction 
			//in order to reuse of operator selection decisions
			BinaryOp ppred0 = null;
			Hop emptyInd = null;
			
			if(selectHop == null) {
				//Step1: compute row/col non-empty indicators 
				ppred0 = new BinaryOp("tmp1", DataType.MATRIX, ValueType.DOUBLE, OpOp2.NOTEQUAL, input, new LiteralOp(0));
				HopRewriteUtils.setOutputBlocksizes(ppred0, brlen, bclen);
				ppred0.refreshSizeInformation();
				ppred0.setForcedExecType(ExecType.SPARK); //always Spark
				HopRewriteUtils.copyLineNumbers(this, ppred0);
				
				emptyInd = ppred0;
				if( !((rmRows && clen == 1) || (!rmRows && rlen==1)) ){
					emptyInd = new AggUnaryOp("tmp2", DataType.MATRIX, ValueType.DOUBLE, AggOp.MAX, rmRows?Direction.Row:Direction.Col, ppred0);
					HopRewriteUtils.setOutputBlocksizes(emptyInd, brlen, bclen);
					emptyInd.refreshSizeInformation();
					emptyInd.setForcedExecType(ExecType.SPARK); //always Spark
					HopRewriteUtils.copyLineNumbers(this, emptyInd);
				}
			} else {
				emptyInd = selectHop;
				HopRewriteUtils.setOutputBlocksizes(emptyInd, brlen, bclen);
				emptyInd.refreshSizeInformation();
				emptyInd.setForcedExecType(ExecType.SPARK); //always Spark
				HopRewriteUtils.copyLineNumbers(this, emptyInd);
			}
			
			//Step 2: compute row offsets for non-empty rows
			Hop cumsumInput = emptyInd;
			if( !rmRows ){
				cumsumInput = HopRewriteUtils.createTranspose(emptyInd);
				HopRewriteUtils.updateHopCharacteristics(cumsumInput, brlen, bclen, this);
			}
		
			UnaryOp cumsum = new UnaryOp("tmp3", DataType.MATRIX, ValueType.DOUBLE, OpOp1.CUMSUM, cumsumInput); 
			HopRewriteUtils.updateHopCharacteristics(cumsum, brlen, bclen, this);
		
			Hop cumsumOutput = cumsum;
			if( !rmRows ){
				cumsumOutput = HopRewriteUtils.createTranspose(cumsum);
				HopRewriteUtils.updateHopCharacteristics(cumsumOutput, brlen, bclen, this);	
			}
			
			Hop maxDim = new AggUnaryOp("tmp4", DataType.SCALAR, ValueType.DOUBLE, AggOp.MAX, Direction.RowCol, cumsumOutput); //alternative: right indexing
			HopRewriteUtils.updateHopCharacteristics(maxDim, brlen, bclen, this);
			
			BinaryOp offsets = new BinaryOp("tmp5", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, cumsumOutput, emptyInd);
			HopRewriteUtils.updateHopCharacteristics(offsets, brlen, bclen, this);
			
			//Step 3: gather non-empty rows/cols into final results 
			Lop linput = input.constructLops();
			Lop loffset = offsets.constructLops();
			Lop lmaxdim = maxDim.constructLops();
			
			HashMap<String, Lop> inMap = new HashMap<String, Lop>();
			inMap.put("target", linput);
			inMap.put("offset", loffset);
			inMap.put("maxdim", lmaxdim);
			inMap.put("margin", inputlops.get("margin"));
		
			if ( !FORCE_DIST_RM_EMPTY && isRemoveEmptyBcSP())
				_bRmEmptyBC = true;
			
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin( inMap, HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et, _bRmEmptyBC);			
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
		
			//Step 4: cleanup hops (allow for garbage collection)
			if(selectHop == null)
				HopRewriteUtils.removeChildReference(ppred0, input);
			
			setLops(pbilop);	
			
			//NOTE: in contrast to mr, replication and aggregation handled instruction-local
		}
	}
	
	/**
	 * 
	 * @param inputlops
	 * @param et
	 * @throws HopsException
	 * @throws LopsException
	 */
	private void constructLopsRExpand(HashMap<String, Lop> inputlops, ExecType et) 
		throws HopsException, LopsException 
	{
		if( et == ExecType.CP || et == ExecType.SPARK )
		{
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops, 
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
			setLops(pbilop);
		}
		else if( et == ExecType.MR )
		{
			ParameterizedBuiltin pbilop = new ParameterizedBuiltin(inputlops, 
					HopsParameterizedBuiltinLops.get(_op), getDataType(), getValueType(), et);
			setOutputDimensions(pbilop);
			setLineNumbers(pbilop);
		
			Group group1 = new Group( pbilop, Group.OperationTypes.Sort, getDataType(), getValueType());
			setOutputDimensions(group1);
			setLineNumbers(group1);
			
			Aggregate finalagg = new Aggregate(group1, Aggregate.OperationTypes.Sum, DataType.MATRIX, getValueType(), ExecType.MR);
			setOutputDimensions(finalagg);
			setLineNumbers(finalagg);
			setLops(finalagg);
		}
	}
	
	
	@Override
	public void printMe() throws HopsException {
		if (LOG.isDebugEnabled()){
			if (getVisited() != VisitStatus.DONE) {
				super.printMe();
				LOG.debug(" " + _op);
			}

			setVisited(VisitStatus.DONE);
		}
	}

	@Override
	protected double computeOutputMemEstimate( long dim1, long dim2, long nnz )
	{	
		double sparsity = OptimizerUtils.getSparsity(dim1, dim2, nnz);
		return OptimizerUtils.estimateSizeExactSparsity(dim1, dim2, sparsity);	
	}
	
	@Override
	protected double computeIntermediateMemEstimate( long dim1, long dim2, long nnz )
	{
		double ret = 0;
		
		if( _op == ParamBuiltinOp.RMEMPTY )
		{ 
			Hop marginHop = getInput().get(_paramIndexMap.get("margin"));
			boolean cols =  marginHop instanceof LiteralOp 
					&& "cols".equals(((LiteralOp)marginHop).getStringValue());
			
			//remove empty has additional internal memory requirements for 
			//computing selection vectors
			if( cols )
			{
				//selection vector: boolean array in the number of columns 
				ret += OptimizerUtils.BIT_SIZE * dim2;
				
				//removeEmpty-cols has additional memory requirements for intermediate 
				//data structures in order to make this a cache-friendly operation.
				ret += OptimizerUtils.INT_SIZE * dim2;				
			}
			else //rows
			{
				//selection vector: boolean array in the number of rows 
				ret += OptimizerUtils.BIT_SIZE * dim1;
			}
		}
		else if( _op == ParamBuiltinOp.REXPAND )
		{
			Hop dir = getInput().get(_paramIndexMap.get("dir"));
			String dirVal = ((LiteralOp)dir).getStringValue();
			if( "rows".equals(dirVal) )
			{
				//rexpand w/ rows direction has additional memory requirements for 
				//intermediate data structures in order to prevent performance issues
				//due to random output row access (to make this cache-friendly)
				//NOTE: bounded by blocksize configuration: at most 12MB
				ret = (OptimizerUtils.DOUBLE_SIZE + OptimizerUtils.INT_SIZE) 
						* Math.min(dim1, 1024*1024);
			}
		}
		
		return ret;
	}
	
	@Override
	protected long[] inferOutputCharacteristics( MemoTable memo )
	{
		//CDF always known because 
		
		long[] ret = null;
	
		Hop input = getInput().get(_paramIndexMap.get("target"));	
		MatrixCharacteristics mc = memo.getAllInputStats(input);

		if( _op == ParamBuiltinOp.GROUPEDAGG ) 
		{
			// Get the number of groups provided as part of aggregate() invocation, whenever available.
			if ( _paramIndexMap.get(Statement.GAGG_NUM_GROUPS) != null ) {
				Hop ngroups = getInput().get(_paramIndexMap.get(Statement.GAGG_NUM_GROUPS));
				if(ngroups != null && ngroups instanceof LiteralOp) {
					long m = HopRewriteUtils.getIntValueSafe((LiteralOp)ngroups);
					return new long[]{m,1,m};
				}
			}
			
			// Output dimensions are completely data dependent. In the worst case, 
			// #groups = #rows in the grouping attribute (e.g., categorical attribute is an ID column, say EmployeeID).
			// In such a case, #rows in the output = #rows in the input. Also, output sparsity is 
			// likely to be 1.0 (e.g., groupedAgg(groups=<a ID column>, fn="count"))
			long m = mc.getRows(); 
			if ( m >= 1 ) {
				ret = new long[]{m, 1, m};
			}
		}
		else if(   _op == ParamBuiltinOp.RMEMPTY ) 
		{ 
			// similar to groupedagg because in the worst-case ouputsize eq inputsize
			// #nnz is exactly the same as in the input but sparsity can be higher if dimensions.
			// change (denser output).
			if ( mc.dimsKnown() ) {
				String margin = "rows";
				Hop marginHop = getInput().get(_paramIndexMap.get("margin"));
				if(    marginHop instanceof LiteralOp 
						&& "cols".equals(((LiteralOp)marginHop).getStringValue()) )
					margin = new String("cols");
				
				MatrixCharacteristics mcSelect = null;
				if (_paramIndexMap.get("select") != null) {
					Hop select = getInput().get(_paramIndexMap.get("select"));	
					mcSelect = memo.getAllInputStats(select);
				}

				long lDim1 = 0, lDim2 = 0;
				if( margin.equals("rows") ) {
					lDim1 = (mcSelect == null ||  !mcSelect.nnzKnown() ) ? mc.getRows(): mcSelect.getNonZeros(); 
					lDim2 = mc.getCols();
				} else {
					lDim1 = mc.getRows();
					lDim2 = (mcSelect == null ||  !mcSelect.nnzKnown() ) ? mc.getCols(): mcSelect.getNonZeros(); 
				}
				ret = new long[]{lDim1, lDim2, mc.getNonZeros()};
			}
		}
		else if(   _op == ParamBuiltinOp.REPLACE ) 
		{ 
			// the worst-case estimate from the input directly propagates to the output 
			// #nnz depends on the replacement pattern and value, same as input if non-zero
			if ( mc.dimsKnown() )
			{
				if( isNonZeroReplaceArguments() )
					ret = new long[]{mc.getRows(), mc.getCols(), mc.getNonZeros()};
				else
					ret = new long[]{mc.getRows(), mc.getCols(), -1};
			}
		}
		else if( _op == ParamBuiltinOp.REXPAND )
		{
			//dimensions are exactly known from input, sparsity unknown but upper bounded by nrow(v)
			//note: cannot infer exact sparsity due to missing cast for outer and potential cutoff for table
			//but very good sparsity estimate possible (number of non-zeros in input)
			Hop max = getInput().get(_paramIndexMap.get("max"));
			Hop dir = getInput().get(_paramIndexMap.get("dir"));
			double maxVal = HopRewriteUtils.getDoubleValueSafe((LiteralOp)max);
			String dirVal = ((LiteralOp)dir).getStringValue();
			if( mc.dimsKnown() )
			{
				long lnnz = mc.nnzKnown() ? mc.getNonZeros() : mc.getRows();
				if( "cols".equals(dirVal) ) { //expand horizontally
					ret = new long[]{mc.getRows(), UtilFunctions.toLong(maxVal), lnnz};
				}
				else if( "rows".equals(dirVal) ){ //expand vertically
					ret = new long[]{UtilFunctions.toLong(maxVal), mc.getRows(), lnnz};
				}	
			}
		}
		
		return ret;
	}
	
	@Override 
	public boolean allowsAllExecTypes()
	{
		return false;
	}
	
	@Override
	protected ExecType optFindExecType() 
		throws HopsException 
	{
		checkAndSetForcedPlatform();

		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		
		if( _etypeForced != null ) 			
		{
			_etype = _etypeForced;	
		}
		else 
		{
			if( _op == ParamBuiltinOp.TRANSFORM )
			{
				// force remote execution type here.
				// At runtime, cp-side transform is triggered for small files.
				return REMOTE;
			}
			
			if ( OptimizerUtils.isMemoryBasedOptLevel() ) {
				_etype = findExecTypeByMemEstimate();
			}
			else if (   _op == ParamBuiltinOp.GROUPEDAGG 
					 && this.getInput().get(0).areDimsBelowThreshold() ) 
			{
				_etype = ExecType.CP;
			}
			else
			{
				_etype = REMOTE;
			}
			
			//check for valid CP dimensions and matrix size
			checkAndSetInvalidCPDimsAndSize();
		}
		
		//mark for recompile (forever)
		if( OptimizerUtils.ALLOW_DYN_RECOMPILATION && !dimsKnown(true) && _etype==REMOTE )
			setRequiresRecompile();
		
		return _etype;
	}
	
	@Override
	public void refreshSizeInformation()
	{
		switch( _op )
		{
			case CDF:
			case INVCDF:	
				//do nothing; CDF is a scalar
				break;
			
			case GROUPEDAGG: { 
				// output dimension dim1 is completely data dependent 
				long ldim1 = -1;
				if ( _paramIndexMap.get(Statement.GAGG_NUM_GROUPS) != null ) {
					Hop ngroups = getInput().get(_paramIndexMap.get(Statement.GAGG_NUM_GROUPS));
					if(ngroups != null && ngroups instanceof LiteralOp) {
						ldim1 = HopRewriteUtils.getIntValueSafe((LiteralOp)ngroups);
					}
				}
				
				setDim1( ldim1 );
				setDim2( 1 );
				break;
			}
			case RMEMPTY: {
				//one output dimension dim1 or dim2 is completely data dependent 
				Hop target = getInput().get(_paramIndexMap.get("target"));
				Hop margin = getInput().get(_paramIndexMap.get("margin"));
				if( margin instanceof LiteralOp ) {
					LiteralOp lmargin = (LiteralOp)margin;
					if( "rows".equals(lmargin.getStringValue()) )
						setDim2( target.getDim2() );
					else if( "cols".equals(lmargin.getStringValue()) )
						setDim1( target.getDim1() );
				}
				setNnz( target.getNnz() );
				break;
			}
			case REPLACE: {
				//dimensions are exactly known from input, sparsity might increase/decrease if pattern/replacement 0 
				Hop target = getInput().get(_paramIndexMap.get("target"));
				setDim1( target.getDim1() );
				setDim2( target.getDim2() );
				if( isNonZeroReplaceArguments() )
					setNnz( target.getNnz() );
				
				break;	
			}
			case REXPAND: {
				//dimensions are exactly known from input, sparsity unknown but upper bounded by nrow(v)
				//note: cannot infer exact sparsity due to missing cast for outer and potential cutoff for table
				Hop target = getInput().get(_paramIndexMap.get("target"));
				Hop max = getInput().get(_paramIndexMap.get("max"));
				Hop dir = getInput().get(_paramIndexMap.get("dir"));
				double maxVal = HopRewriteUtils.getDoubleValueSafe((LiteralOp)max);
				String dirVal = ((LiteralOp)dir).getStringValue();
				
				if( "cols".equals(dirVal) ) { //expand horizontally
					setDim1(target.getDim1());
					setDim2(UtilFunctions.toLong(maxVal));
				}
				else if( "rows".equals(dirVal) ){ //expand vertically
					setDim1(UtilFunctions.toLong(maxVal));
					setDim2(target.getDim1());
				}
				
				break;	
			}
			default:
				//do nothing
				break;
		}
	}
	
	@Override
	@SuppressWarnings("unchecked")
	public Object clone() throws CloneNotSupportedException 
	{
		ParameterizedBuiltinOp ret = new ParameterizedBuiltinOp();	
		
		//copy generic attributes
		ret.clone(this, false);
		
		//copy specific attributes
		ret._op = _op;
		ret._outputEmptyBlocks = _outputEmptyBlocks;
		ret._outputPermutationMatrix = _outputPermutationMatrix;
		ret._paramIndexMap = (HashMap<String, Integer>) _paramIndexMap.clone();
		//note: no deep cp of params since read-only 
		
		return ret;
	}
	
	@Override
	public boolean compare( Hop that )
	{
		if( !(that instanceof ParameterizedBuiltinOp) )
			return false;
		
		ParameterizedBuiltinOp that2 = (ParameterizedBuiltinOp)that;	
		boolean ret = (_op == that2._op
					  && _paramIndexMap!=null && that2._paramIndexMap!=null
					  && _paramIndexMap.size() == that2._paramIndexMap.size()
					  && _outputEmptyBlocks == that2._outputEmptyBlocks
					  && _outputPermutationMatrix == that2._outputPermutationMatrix );
		if( ret )
		{
			for( Entry<String,Integer> e : _paramIndexMap.entrySet() )
			{
				String key1 = e.getKey();
				int pos1 = e.getValue();
				int pos2 = that2._paramIndexMap.get(key1);
				ret &= (   that2.getInput().get(pos2)!=null
					    && getInput().get(pos1) == that2.getInput().get(pos2) );
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	@Override
	public boolean isTransposeSafe()
	{
		boolean ret = false;
		
		try
		{
			if( _op == ParamBuiltinOp.GROUPEDAGG )
			{
				int ix = _paramIndexMap.get(Statement.GAGG_FN);
				Hop fnHop = getInput().get(ix);
				ret = (fnHop instanceof LiteralOp && Statement.GAGG_FN_SUM.equals(((LiteralOp)fnHop).getStringValue()) );
			}
		}
		catch(Exception ex) {
			//silent false
			LOG.warn("Check for transpose-safeness failed, continue assuming false.", ex);
		}
		
		return ret;	
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isCountFunction()
	{
		boolean ret = false;
		
		try
		{
			if( _op == ParamBuiltinOp.GROUPEDAGG )
			{
				int ix = _paramIndexMap.get(Statement.GAGG_FN);
				Hop fnHop = getInput().get(ix);
				ret = (fnHop instanceof LiteralOp && Statement.GAGG_FN_COUNT.equals(((LiteralOp)fnHop).getStringValue()) );
			}
		}
		catch(Exception ex){
			//silent false
			LOG.warn("Check for count function failed, continue assuming false.", ex);
		}
		
		return ret;
	}
	
	/**
	 * Only applies to REPLACE.
	 * @return
	 */
	private boolean isNonZeroReplaceArguments()
	{
		boolean ret = false;
		try 
		{
			Hop pattern = getInput().get(_paramIndexMap.get("pattern"));
			Hop replace = getInput().get(_paramIndexMap.get("replacement"));
			if( pattern instanceof LiteralOp && ((LiteralOp)pattern).getDoubleValue()!=0d &&
			    replace instanceof LiteralOp && ((LiteralOp)replace).getDoubleValue()!=0d )
			{
				ret = true;
			}
		}
		catch(Exception ex) 
		{
			LOG.warn(ex.getMessage());	
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @return
	 */
	public boolean isTargetDiagInput()
	{
		Hop targetHop = getTargetHop();
		
		//input vector (guarantees diagV2M), implies remove rows
		return (   targetHop instanceof ReorgOp 
				&& ((ReorgOp)targetHop).getOp()==ReOrgOp.DIAG 
				&& targetHop.getInput().get(0).getDim2() == 1 ); 
	}

	/**
	 * This will check if there is sufficient memory locally (twice the size of second matrix, for original and sort data), and remotely (size of second matrix (sorted data)).  
	 * @return
	 */
	private boolean isRemoveEmptyBcSP()	// TODO find if 2 x size needed. 
	{
		boolean ret = false;
		Hop input = getInput().get(0);
		
		//note: both cases (partitioned matrix, and sorted double array), require to
		//fit the broadcast twice into the local memory budget. Also, the memory 
		//constraint only needs to take the rhs into account because the output is 
		//guaranteed to be an aggregate of <=16KB
		
		double size = input.dimsKnown() ? 
				OptimizerUtils.estimateSize(input.getDim1(), 1) : //dims known and estimate fits
					input.getOutputMemEstimate();                 //dims unknown but worst-case estimate fits
		
		if( OptimizerUtils.checkSparkBroadcastMemoryBudget(size) ) {
			ret = true;
		}
		
		return ret;
	}	
	
}
