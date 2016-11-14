package org.apache.sysml.api.ml.evaluation;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.shared.HasLabelCol;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;

import scala.Tuple2;

public class SurvivalAnalysisEvaluator extends Evaluator implements HasLabelCol {

	private static final long serialVersionUID = 1132742678204911537L;

	// private Param<String> metricName = new Param<String>(this,
	// "metricName", "Metric name in evaluation");
	private Param<String> labelCol = new Param<String>(this, "labelCol", "Name of the label column");
	private String labelColName = "";

	public SurvivalAnalysisEvaluator() {
		setLabelCol("label");
	}

	@Override
	public String uid() {
		return Long.toString(serialVersionUID);
	}

	@Override
	public Evaluator copy(ParamMap paramMap) {
		return defaultCopy(paramMap);
	}

	public SurvivalAnalysisEvaluator setLabelCol(String value) {
		labelColName = value;
		return (SurvivalAnalysisEvaluator) setDefault(labelCol, value);
	}

	@Override
	public String getLabelCol() {
		return labelColName;
	}

	@Override
	public Param<String> labelCol() {
		return labelCol;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasLabelCol$_setter_$labelCol_$eq(Param arg0) {

	}

	@Override
	public double evaluate(DataFrame dataset) {
		List<Row> input = dataset.select(getLabelCol(), "Risk").rdd().toJavaRDD().collect();
		List<Tuple2<Double, Double>> pair = new ArrayList<Tuple2<Double, Double>>();
		long agree = 0, tied = 0, disagree = 0;

		for (Row entry : input) {
			DenseVector dv = (DenseVector) entry.get(0);
			pair.add(new Tuple2<Double, Double>(dv.apply(0), (Double) entry.get(1)));
		}

		int size = pair.size();
		for (int i = 0; i < size; i++) {
			for (int j = i + 1; j < size; j++) {
				double prevTimestamp = pair.get(i)._1;
				double currTimestamp = pair.get(j)._1;
				double prevRisk = pair.get(i)._2;
				double currRisk = pair.get(j)._2;

				if ((prevTimestamp < currTimestamp) && (prevRisk > currRisk))
					agree++;
				else if ((prevTimestamp > currTimestamp) && (prevRisk < currRisk))
					disagree++;
				else
					tied++;
			}
		}
		double result = (double) (agree + tied / 2) / (agree + disagree + tied);

		return result;
	}
}
