package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import org.apache.sysml.parser.Expression;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MetaDataFormat;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * Disjoint_Round_Robin data partitioner:
 * <p>
 * for each worker, use a permutation multiply
 * or simpler a removeEmpty such as removeEmpty
 * (target=X, margin=rows, select=(seq(1,nrow(X))%%k)==id)
 */
public class DataPartitionerDRR extends DataPartitioner {
	@Override
	public List<MatrixObject> doPartition(int k, MatrixObject mo) {
		return IntStream.range(0, k).mapToObj(i -> removeEmpty(mo, k, i)).collect(Collectors.toList());
	}

	private MatrixObject removeEmpty(MatrixObject mo, int k, int workerId) {
		MatrixObject result = new MatrixObject(Expression.ValueType.DOUBLE, null,
				new MetaDataFormat(new MatrixCharacteristics(-1, -1, -1, -1), OutputInfo.BinaryBlockOutputInfo,
						InputInfo.BinaryBlockInputInfo));
		MatrixBlock tmp = mo.acquireRead();
		double[] data = LongStream.range(0, mo.getNumRows()).mapToDouble(l -> l % k == workerId ? 0 : 1).toArray();
		MatrixBlock select = DataConverter.convertToMatrixBlock(data, true);
		MatrixBlock resultMB = tmp.removeEmptyOperations(new MatrixBlock(), true, true, select);
		mo.release();
		result.acquireModify(resultMB);
		result.release();
		result.enableCleanup(false);
		return result;
	}
}
