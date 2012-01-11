package dml.hops;

import dml.hops.Hops.AggOp;
import dml.hops.Hops.DataOpTypes;
import dml.hops.Hops.OpOp2;
import dml.parser.Expression.DataType;
import dml.parser.Expression.ValueType;
import dml.utils.HopsException;

public class TestDriver {

	static void test1() throws HopsException {

		/*
		 * Example:
		 * 
		 * result = colSum(T((((a %*% b) * c) * 10)));
		 */

		Hops a = new DataOp("A", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "A.data",-1,-1,-1,-1);
		Hops b = new DataOp("B", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "B.data",-1,-1,-1,-1);
		Hops c = new DataOp("C", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "C.data",-1,-1,-1,-1);
		
		Hops t1;

		t1 = new AggBinaryOp("", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, AggOp.SUM, a, b);
		t1 = new BinaryOp("", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, t1, c);
	
		t1.printMe();
	};
	static void test2() throws HopsException {

		/*
		 * Example:
		 * 
		 * A = B %*% C; D = A + 20; E = A * 50;
		 */

		Hops b = new DataOp("B", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "B.data",-1,-1,-1,-1);
		Hops c = new DataOp("C", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "C.data",-1,-1,-1,-1);

		b = new DataOp("B", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "B.data",-1,-1,-1,-1);
		c = new DataOp("C", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "C.data",-1,-1,-1,-1);

		Hops t1 = new AggBinaryOp("A", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, AggOp.SUM, b, c);
		Hops t2 = new BinaryOp("D", DataType.MATRIX, ValueType.DOUBLE, OpOp2.PLUS, t1, new LiteralOp("20", 20));
		Hops t3 = new BinaryOp("E", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, t1, new LiteralOp("A", 50));

		t2.printMe();
		t3.printMe();

	}
	static void test3() throws HopsException {
		/*
		 * Example (example0.1.R)
		 * 
		 * C = A %*% B write (C)
		 */

		Hops a = new DataOp("A", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "A.data",-1,-1,-1,-1);
		Hops b = new DataOp("B", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "B.data",-1,-1,-1,-1);

		a = new DataOp("A", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "A.data",-1,-1,-1,-1);
		b = new DataOp("B", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "B.data",-1,-1,-1,-1);

		Hops t1 = new AggBinaryOp("C", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, AggOp.SUM, a, b);
		Hops t2 = new DataOp("C", DataType.MATRIX, ValueType.DOUBLE, t1, DataOpTypes.PERSISTENTWRITE, "C.data");

		t1 = new AggBinaryOp("C", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, AggOp.SUM, a, b);
		t2 = new DataOp("C", DataType.MATRIX, ValueType.DOUBLE, t1, DataOpTypes.PERSISTENTWRITE, "C.data");


		t2.printMe();
		t2.constructLops().printMe();

	}
	static void test4() throws HopsException {


		//t2.printMe();
		//t2.constructLops().printMe();

	}
	static void test5() throws HopsException {

		/*
		 * Example (example0.3.R):
		 * 
		 * A = read (A); B = t(A); C = A + B; D = C * 5; write (D);
		 */


		Hops a = new DataOp("A", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "A.data",-1,-1,-1,-1);

		a = new DataOp("A", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "A.data",-1,-1,-1,-1);

		a.set_dim1(200);
		a.set_dim2(1000);
		Hops b = new ReorgOp("B", DataType.MATRIX, ValueType.DOUBLE, Hops.ReorgOp.TRANSPOSE, a);
		b.set_dim1(1000);
		b.set_dim2(200);
	//	Hops c = new BinaryOp("C", DataType.MATRIX, ValueType.DOUBLE, OpOp2.PLUS, a, b);
	//	LiteralOp l = new LiteralOp("L", 5);
//		Hops d = new UnaryOp("D", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, c, l, Position.after);

	//	Hops t1 = new DataOp("", DataType.MATRIX, ValueType.DOUBLE, d, DataOpTypes.PERSISTENTWRITE, "D.data");

		//t1 = new DataOp("", DataType.MATRIX, ValueType.DOUBLE, d, DataOpTypes.PERSISTENTWRITE, "D.data");


//		t1.printMe();
	//	t1.constructLops().printMe();
	}
	static void test6() throws HopsException {
		
		// i = 10.0 + 5.0
		// i = i + 20.0
		// w = ReadMM ("W.data")
		// w = w * i
		
		
		LiteralOp i10 = new LiteralOp("i10", 10.0);
		LiteralOp i5 = new LiteralOp("i5", 5.0);
		Hops i = new BinaryOp("i", DataType.SCALAR, ValueType.DOUBLE, OpOp2.PLUS, i10, i5);
		LiteralOp i20 = new LiteralOp("i20", 20.0);
		
		i = new BinaryOp("i", DataType.SCALAR, ValueType.DOUBLE, OpOp2.PLUS, i, i20);
		
		Hops w = new DataOp("w", DataType.MATRIX, ValueType.DOUBLE, DataOpTypes.PERSISTENTREAD, "W.data", 200, 1000, -1,-1);
		w = new BinaryOp("w", DataType.MATRIX, ValueType.DOUBLE, OpOp2.MULT, w, i);	
		
		w.printMe();
	}
	
	public static void main(String[] args) throws HopsException {
		
		// test1();
		// test2();
		// test3();
		// test4();
		// test5();
		test6();

	}

}
