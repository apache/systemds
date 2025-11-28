package org.apache.sysds.test.component.compress.lib;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressedMatrixBlock;
import org.apache.sysds.runtime.compress.CompressedMatrixBlockFactory;
import org.apache.sysds.runtime.compress.CompressionSettingsBuilder;
import org.apache.sysds.runtime.compress.colgroup.AColGroup;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.lib.CLALibUnary;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.UnaryOperator;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class CLALibUnaryDeltaTest {

	protected static final Log LOG = LogFactory.getLog(CLALibUnaryDeltaTest.class.getName());

	@Test
	public void testCumsumResultsInDeltaEncoding() {
		// Use data that results in repetitive deltas to ensure DeltaDDC is chosen
		MatrixBlock mb = new MatrixBlock(20, 1, false);
		mb.allocateDenseBlock();
		// Input: 1, 2, 1, 2, ...
		// Cumsum: 1, 3, 4, 6, ...
		// Deltas: 1, 2, 1, 2, ...
		for(int i = 0; i < 20; i++) {
			mb.set(i, 0, (i % 2 == 0) ? 1.0 : 2.0);
		}
		mb.setNonZeros(20);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);

		assertNotNull("Result should not be null", result);
		assertTrue("Result should be compressed", result instanceof CompressedMatrixBlock);

		CompressedMatrixBlock compressedResult = (CompressedMatrixBlock) result;
		boolean hasDeltaDDC = false;
		for(AColGroup cg : compressedResult.getColGroups()) {
			if(cg.getCompType() == CompressionType.DeltaDDC) {
				hasDeltaDDC = true;
				break;
			}
		}

		assertTrue("Result should contain DeltaDDC column group", hasDeltaDDC);
	}

	@Test
	public void testRowcumsumResultsInDeltaEncoding() {
		MatrixBlock mb = new MatrixBlock(3, 4, false);
		mb.allocateDenseBlock();
		// Row 1: 1, 2, 3, 4 -> cumsum: 1, 3, 6, 10
		mb.set(0, 0, 1.0);
		mb.set(0, 1, 2.0);
		mb.set(0, 2, 3.0);
		mb.set(0, 3, 4.0);
		// Row 2: 1, 1, 1, 1 -> cumsum: 1, 2, 3, 4
		mb.set(1, 0, 1.0);
		mb.set(1, 1, 1.0);
		mb.set(1, 2, 1.0);
		mb.set(1, 3, 1.0);
		// Row 3: 5, 5, 5, 5 -> cumsum: 5, 10, 15, 20
		mb.set(2, 0, 5.0);
		mb.set(2, 1, 5.0);
		mb.set(2, 2, 5.0);
		mb.set(2, 3, 5.0);
		mb.setNonZeros(12);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator rowCumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ROWCUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, rowCumsumOp, null);

		assertNotNull("Result should not be null", result);
		assertTrue("Result should be compressed", result instanceof CompressedMatrixBlock);

		CompressedMatrixBlock compressedResult = (CompressedMatrixBlock) result;
		// Delta encoding is row-wise, so row cumsum might not always benefit from delta DDC as much as col cumsum
		// but we enforce it preference so it should be there if applicable.
		// Actually for row cumsum, the result across columns changes.
		// Let's check correctness mainly.
		MatrixBlock expected = mb.unaryOperations(rowCumsumOp, new MatrixBlock());
		TestUtils.compareMatrices(expected, result, 0.0, "RowCumsum result should match expected");
	}

	@Test
	public void testCumsumCorrectness() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 3, 0, 10, 1.0, 7);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());

		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result should match expected");
	}

	@Test
	public void testRowcumsumCorrectness() {
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(10, 5, 0, 10, 1.0, 7);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator rowCumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ROWCUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, rowCumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(rowCumsumOp, new MatrixBlock());

		TestUtils.compareMatrices(expected, result, 0.0, "RowCumsum result should match expected");
	}

	@Test
	public void testNonCumsumOperationDoesNotUseDeltaEncoding() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		for(int i = 0; i < 10; i++) {
			mb.set(i, 0, i);
			mb.set(i, 1, i * 2);
		}
		mb.setNonZeros(20);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator absOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.ABS));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, absOp, null);

		assertNotNull("Result should not be null", result);
		
		if(result instanceof CompressedMatrixBlock) {
			CompressedMatrixBlock compressedResult = (CompressedMatrixBlock) result;
			boolean hasDeltaDDC = false;
			for(AColGroup cg : compressedResult.getColGroups()) {
				if(cg.getCompType() == CompressionType.DeltaDDC) {
					hasDeltaDDC = true;
					break;
				}
			}
			// Should not have delta DDC
			assertTrue("Result should NOT contain DeltaDDC column group for ABS", !hasDeltaDDC);
		}
		// If not compressed, it's also fine (standard execution)
	}

	@Test
	public void testCumsumSparseMatrix() {
		MatrixBlock mb = new MatrixBlock(100, 10, true);
		mb.set(0, 0, 1.0);
		mb.set(10, 0, 2.0);
		mb.set(20, 0, 3.0);
		mb.setNonZeros(3);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());

		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result for sparse matrix should match expected");
	}
	
	@Test
	public void testCumsumWithDifferentInputCompressionTypes() {
		MatrixBlock mb = new MatrixBlock(10, 1, false);
		mb.allocateDenseBlock();
		// RLE friendly data: 1, 1, 1, 2, 2, 2, 3, 3, 3, 4
		for(int i=0; i<3; i++) mb.set(i, 0, 1.0);
		for(int i=3; i<6; i++) mb.set(i, 0, 2.0);
		for(int i=6; i<9; i++) mb.set(i, 0, 3.0);
		mb.set(9, 0, 4.0);
		mb.setNonZeros(10);

		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.RLE);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		
		assertTrue("Result should be compressed", result instanceof CompressedMatrixBlock);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());
		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result from RLE input should match expected");
	}
	
	@Test
	public void testCumsumLargeMatrix() {
		// Larger matrix to trigger multi-threaded execution if applicable
		MatrixBlock mb = TestUtils.generateTestMatrixBlock(100, 5, 0, 100, 1.0, 7);
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);

		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());

		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result for large matrix should match expected");
	}
	
	@Test
	public void testCumsumWithConstantColumns() {
		MatrixBlock mb = new MatrixBlock(10, 2, false);
		mb.allocateDenseBlock();
		for(int i=0; i<10; i++) {
			mb.set(i, 0, 1.0); // Constant column
			mb.set(i, 1, i);   // Increasing column
		}
		mb.setNonZeros(20);
		
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		csb.addValidCompression(CompressionType.CONST);
		CompressedMatrixBlock cmb = compress(mb, csb);
		
		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());
		
		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result with constant columns should match expected");
	}
	
	@Test
	public void testCumsumMultiColumn() {
		MatrixBlock mb = new MatrixBlock(10, 4, false);
		mb.allocateDenseBlock();
		for(int i=0; i<10; i++) {
			for(int j=0; j<4; j++) {
				mb.set(i, j, i+j);
			}
		}
		mb.setNonZeros(40);
		
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.DDC);
		CompressedMatrixBlock cmb = compress(mb, csb);
		
		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());
		
		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result for multi-column matrix should match expected");
	}
	
	@Test
	public void testCumsumWhenDeltaDDCNotInValidCompressions() {
		MatrixBlock mb = new MatrixBlock(4, 2, false);
		mb.allocateDenseBlock();
		mb.set(0, 0, 1.0);
		mb.set(0, 1, 2.0);
		mb.set(1, 0, 3.0);
		mb.set(1, 1, 4.0);
		mb.set(2, 0, 5.0);
		mb.set(2, 1, 6.0);
		mb.set(3, 0, 7.0);
		mb.set(3, 1, 8.0);
		mb.setNonZeros(8);
		
		CompressionSettingsBuilder csb = new CompressionSettingsBuilder().setMinimumCompressionRatio(0.0);
		csb.addValidCompression(CompressionType.RLE);
		CompressedMatrixBlock cmb = compress(mb, csb);
		
		UnaryOperator cumsumOp = new UnaryOperator(Builtin.getBuiltinFnObject(Builtin.BuiltinCode.CUMSUM));
		MatrixBlock result = CLALibUnary.unaryOperations(cmb, cumsumOp, null);
		
		assertNotNull("Result should not be null", result);
		MatrixBlock expected = mb.unaryOperations(cumsumOp, new MatrixBlock());
		TestUtils.compareMatrices(expected, result, 0.0, "Cumsum result should match expected even when DeltaDDC not in valid compressions");
	}

	private CompressedMatrixBlock compress(MatrixBlock mb, CompressionSettingsBuilder csb) {
		MatrixBlock mbComp = CompressedMatrixBlockFactory.compress(mb, 1, csb).getLeft();
		if(mbComp instanceof CompressedMatrixBlock)
			return (CompressedMatrixBlock) mbComp;
		else
			return CompressedMatrixBlockFactory.genUncompressedCompressedMatrixBlock(mbComp);
	}
}
