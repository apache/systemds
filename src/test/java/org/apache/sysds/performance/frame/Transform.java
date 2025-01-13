package org.apache.sysds.performance.frame;

import java.util.Arrays;

import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.performance.compression.APerfTest;
import org.apache.sysds.performance.generators.ConstFrame;
import org.apache.sysds.performance.generators.IGenerate;
import org.apache.sysds.runtime.frame.data.FrameBlock;
import org.apache.sysds.runtime.frame.data.columns.Array;
import org.apache.sysds.runtime.transform.encode.EncoderFactory;
import org.apache.sysds.runtime.transform.encode.MultiColumnEncoder;
import org.apache.sysds.test.TestUtils;

public class Transform extends APerfTest<Object, FrameBlock> {

	private final int k;
	private final String spec;

	public Transform(int N, IGenerate<FrameBlock> gen, int k, String spec) {
		super(N, gen);
		this.k = k;
		this.spec = spec;
		FrameBlock in = gen.take();
		System.out.println("Transform Encode Perf: rows: " + in.getNumRows() + " schema:" + Arrays.toString(in.getSchema()));
		System.out.println(spec);
	}

	public void run() throws Exception {
		execute(() -> te(), () -> clear(), "Normal");
		execute(() -> tec(), () -> clear(), "Compressed");
		execute(() -> te(), () -> clear(), "Normal");
		execute(() -> tec(), () -> clear(), "Compressed");
	}

	private void te(){
		FrameBlock in = gen.take();
		MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, in.getNumColumns());
		enc.encode(in, k);
		ret.add(null);
	}

	private void tec(){
		FrameBlock in = gen.take();
		MultiColumnEncoder enc = EncoderFactory.createEncoder(spec, in.getNumColumns());
		enc.encode(in, k, true);
		ret.add(null);
	}

	private void clear(){
		clearRDCCache(gen.take());
	}

	@Override
	protected String makeResString() {
		return "";
	}


		/**
	 * Forcefully clear recode cache of underlying arrays
	 */
	public void clearRDCCache(FrameBlock f){
		for(Array<?> a : f.getColumns())
			a.setCache(null);
	}


	public static void main(String[] args) throws Exception {
		for(int i = 1; i < 100; i *= 10){

			FrameBlock in = TestUtils.generateRandomFrameBlock(100000 * i , new ValueType[]{ValueType.UINT4}, 32);
			System.out.println(Arrays.toString(in.getColumnNames()));
			ConstFrame gen = new ConstFrame(in);
			// passthrough
			new Transform(300, gen, 16, "{}").run();
			new Transform(300, gen, 16, "{ids:true, recode:[1]}").run();
			new Transform(300, gen, 16, "{ids:true, bin:[{id:1, method:equi-width, numbins:4}]}").run();
			new Transform(300, gen, 16, "{ids:true, bin:[{id:1, method:equi-width, numbins:4}], dummycode:[1]}").run();
			new Transform(300, gen, 16, "{ids:true, hash:[1], K:10}").run();
			new Transform(300, gen, 16, "{ids:true, hash:[1], K:10, dummycode:[1]}").run();
		}

		System.exit(0); // forcefully stop.
	}

}
