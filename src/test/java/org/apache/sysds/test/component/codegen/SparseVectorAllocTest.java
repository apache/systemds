package org.apache.sysds.test.component.codegen;

import org.apache.sysds.runtime.codegen.LibSpoofPrimitives;
import org.apache.sysds.runtime.data.SparseRowVector;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

/**
 * This is the component test for the ring buffer used in LibSpoofPrimitives.java,
 * that allocates a vector with a certain size.
 * Every allocation method is tested to achieve the needed coverage.
 */
public class SparseVectorAllocTest extends AutomatedTestBase
{
	double[] val1 = new double[]{1.5, 5.7, 9.1, 3.7, 5.3};
	double[] val2 = new double[]{9.6, 7.1, 2.7};
	int[] indexes1 = new int[]{3, 7, 14, 20, 81};
	int[] indexes2 = new int[]{20, 30, 90};

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
	}

	@Test
	public void testBasicAllocationSameLen() {
		testBasicSparseVectorAllocation(1, 10, 10);
	}

	@Test
	public void testBasicAllocationLongerExp() {
		testBasicSparseVectorAllocation(1, 10, 15);
	}

	@Test
	public void testBasicAllocationShorterExp() {
		testBasicSparseVectorAllocation(1, 10, 7);
	}

	@Test
	public void testVectorReuse1() {
		testBufferReuse(3, 10, 5, 5);
	}

	@Test
	public void testVectorReuse2() {
		testBufferReuse(3, 10, -1, 5);
	}

	/** tests the allocation of an empty vector
	 * @param numVectors number of vectors that should be pre-allocated
	 * @param len the length of the vector
	 * @param expLen the expected length of the allocated vector
	 */
	public void testBasicSparseVectorAllocation(int numVectors, int len, int expLen) {
		//test the basic allocation of an empty vector
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len, -1);
		SparseRowVector sparseVec = LibSpoofPrimitives.allocSparseVector(expLen);

		Assert.assertTrue("Vector capacity should be initialized correctly", expLen <= sparseVec.capacity());
		Assert.assertEquals("Vector size should be initialized with 0", 0, sparseVec.size());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

	/** tests the allocation of a vector that is reused multiple times
	 * @param numVectors number of vectors that should be pre-allocated
	 * @param len1 length of the first vector
	 * @param len2 length of the second vector
	 * @param expLen expected length of allocated vector
	 */
	public void testBufferReuse(int numVectors, int len1, int len2, int expLen) {
		//test the reuse of the vectors in the ring buffer
		LibSpoofPrimitives.setupSparseThreadLocalMemory(numVectors, len1, len2);

		//allocate first vector
		SparseRowVector vec1 = LibSpoofPrimitives.allocSparseVector(expLen);
		vec1.set(0, 1.0);
		vec1.set(2, 2.0);

		//allocate second vector
		SparseRowVector vec2 = LibSpoofPrimitives.allocSparseVector(expLen);

		Assert.assertEquals("Reused vector should be reset to size 0", 0, vec2.size());

		for(int j = 0; j < vec2.size(); j++) {
			vec2.set(vec2.indexes()[j], vec2.get(vec2.indexes()[j]) * 32);
		}

		SparseRowVector vec3 = LibSpoofPrimitives.allocSparseVector(expLen);

		Assert.assertEquals("Reused vector should be reset to size 0", 0, vec3.size());

		SparseRowVector vec4 = LibSpoofPrimitives.allocSparseVector(expLen);

		for(int j = 0; j < vec4.size(); j++) {
			vec4.set(vec4.indexes()[j], vec4.get(vec3.indexes()[j]) * 32);
		}

		Assert.assertEquals("Reused vector should be reset to size 0", 0, vec4.size());

		SparseRowVector vec5 = LibSpoofPrimitives.allocSparseVector(expLen);

		for(int j = 0; j < vec5.size(); j++) {
			vec2.set(vec5.indexes()[j], vec5.get(vec5.indexes()[j]) * 32);
		}

		Assert.assertEquals("Reused vector should be reset to size 0", 0, vec5.size());

		LibSpoofPrimitives.cleanupSparseThreadLocalMemory();
	}

}
