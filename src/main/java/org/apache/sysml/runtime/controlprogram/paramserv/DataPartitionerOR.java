package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.functionobjects.Multiply;
import org.apache.sysml.runtime.functionobjects.Plus;
import org.apache.sysml.runtime.instructions.cp.AggregateBinaryCPInstruction;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;
import org.apache.sysml.runtime.matrix.operators.AggregateOperator;

/**
 * Data partitioner Overlap_Reshuffle:
 * for each worker, use a new permutation multiply P %*% X,
 * where P is constructed for example with P=table(seq(1,nrow(X),sample(nrow(X), nrow(X))))
 */
public class DataPartitionerOR extends DataPartitioner {
	@Override
	public List<MatrixObject> doPartition(int k, MatrixObject mo) {
		ExecutionContext ec = ExecutionContextFactory.createContext();
		ec.setVariable("data", mo);

		AggregateOperator agg = new AggregateOperator(0, Plus.getPlusFnObject());
		AggregateBinaryOperator aggbin = new AggregateBinaryOperator(Multiply.getMultiplyFnObject(), agg, 1);
		AggregateBinaryCPInstruction multiInst = new AggregateBinaryCPInstruction(aggbin,
				new CPOperand("permutation", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				new CPOperand("data"), new CPOperand("result", Expression.ValueType.DOUBLE, Expression.DataType.MATRIX),
				"ba+*", "permutation multiply");

		return IntStream.range(0, k).mapToObj(i -> {
			// Generate the permutation
			MatrixObject permutation = ParamservUtils.generatePermutation(mo, ec);
			MatrixObject result = ParamservUtils.newMatrixObject();
			ec.setVariable("result", result);
			multiInst.processInstruction(ec);
			return result;
		}).collect(Collectors.toList());
	}
}
