package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.sysml.hops.Hop;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.CtableCPInstruction;
import org.apache.sysml.runtime.instructions.cp.DataGenCPInstruction;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;
import org.apache.sysml.runtime.util.DataConverter;

public class DataPartitionerDR extends DataPartitioner {

	@Override
	public List<MatrixObject> doPartition(int k, MatrixObject mo) {
		ExecutionContext ec = ExecutionContextFactory.createContext();

		// Create the sequence
		double[] data = LongStream.range(1, mo.getNumRows() + 1).mapToDouble(l -> l).toArray();
		MatrixBlock seqMB = DataConverter.convertToMatrixBlock(data, true);
		MatrixObject seq = ParamservUtils.newMatrixObject();
		seq.acquireModify(seqMB);
		seq.release();
		ec.setVariable("seq", seq);

		// Generate a sample
		DataGenCPInstruction sampleInst = new DataGenCPInstruction(null, Hop.DataGenMethod.SAMPLE, null,
				new CPOperand("sample", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand(String.valueOf(mo.getNumRows()), Expression.ValueType.INT, Expression.DataType.SCALAR,
						true), new CPOperand("1", Expression.ValueType.INT, Expression.DataType.SCALAR, true),
				(int) mo.getNumRowsPerBlock(), (int) mo.getNumColumnsPerBlock(), mo.getNumRows(), false, -1,
				Hop.DataGenMethod.SAMPLE.name().toLowerCase(), "sample");
		ec.setVariable("sample", ParamservUtils.newMatrixObject());
		sampleInst.processInstruction(ec);

		// Combine the sequence and sample as a table
		CtableCPInstruction tableInst = new CtableCPInstruction(
				new CPOperand("seq", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("sample", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("1.0", Expression.ValueType.DOUBLE, Expression.DataType.SCALAR, true),
				new CPOperand("permutation", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX), "-1", true, "-1",
				true, true, false, "ctableexpand", "table");
		ec.setVariable("permutation", ParamservUtils.newMatrixObject());
		tableInst.processInstruction(ec);

		// Slice the original matrix and make data partition by permutation multiply
		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, 1);
		AggregateBinaryCPInstruction multiInst = new AggregateBinaryCPInstruction(aggbin,
				new CPOperand("permutation", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("data"), new CPOperand("result", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				"ba+*", "permutation multiply");
		ec.setVariable("data", mo);

		MatrixObject permutation = (MatrixObject) ec.getVariable("permutation");
		int batchSize = (int) Math.ceil(mo.getNumRows() / k);
		return IntStream.range(0, k).mapToObj(i -> {
			long begin = i * batchSize + 1;
			long end = Math.min(begin + batchSize, mo.getNumRows());
			MatrixObject partialMO = ParamservUtils.sliceMatrix(permutation, begin, end);
			ec.setVariable("permutation", partialMO);
			MatrixObject result = ParamservUtils.newMatrixObject();
			ec.setVariable("result", result);
			multiInst.processInstruction(ec);
			ParamservUtils.cleanupData(partialMO);
			return result;
		}).collect(Collectors.toList());
	}
}
