package org.apache.sysds.performance.matrix;

import java.util.Random;

import org.apache.sysds.performance.TimingUtils;
import org.apache.sysds.runtime.data.SparseBlockMCSR;

public class SparseAppend {

	public SparseAppend(String[] args) {
		// Ignore args 1 it is containing the Id to run this test.

		final int rep = Integer.parseInt(args[1]);
		final int size = Integer.parseInt(args[2]);
		final Random r = new Random(42);

		double[] times;
		// Warmup 
		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, r.nextDouble());
			}
		}, rep);

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, r.nextDouble());
			}
		}, rep);

		System.out.println("Append all dense:          " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on empty:  " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on Scalar: " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				sb.append(i % 100, i, 0);
			}
		}, rep);

		System.out.println("Append all zero on Array:  " + TimingUtils.stats(times));

		times = TimingUtils.time(() -> {
			SparseBlockMCSR sb = new SparseBlockMCSR(100);
			for(int i = 0; i < 100; i ++){
				sb.append(i, i, 1);
				sb.append(i, i, 1);
			}
			for(int i = 0; i < size; i++) {
				double d = r.nextDouble();
				d = d > 0.5 ? d : 0;
				sb.append(i % 100, i, d);
			}
		}, rep);

		System.out.println("Append half zero on Array: " + TimingUtils.stats(times));
	}

	public static void main(String[] args) {
		new SparseAppend(new String[] {"1004", "10000", "10000"});
	}
}
