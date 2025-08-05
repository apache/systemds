/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.common;

import java.util.EnumSet;
import java.util.HashMap;

import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ReturnType;

/**
 * Enum to represent all builtin functions in the default name space.
 * Each function is either native or implemented by a DML script. In
 * case of DML script, these functions are loaded during parsing. As
 * always, user-defined DML-bodied functions take precedence over all
 * builtin functions.
 *
 * To add a new builtin script function, simply add the definition here
 * as well as a dml file in scripts/builtin with a matching name. On
 * building SystemDS, these scripts are packaged into the jar as well.
 */
public enum Builtins {
	// Builtin functions without parameters
	ABSTAIN("abstain", true),
	ABS("abs", false),
	ACOS("acos", false),
	ADASYN("adasyn", true),
	ALS("als", true),
	ALS_CG("alsCG", true),
	ALS_DS("alsDS", true),
	ALS_PREDICT("alsPredict", true),
	ALS_TOPK_PREDICT("alsTopkPredict", true),
	AMPUTE("ampute", true),
	APPLY_PIPELINE("apply_pipeline", true),
	APPLY_SCHEMA("applySchema", false),
	ARIMA("arima", true),
	ASIN("asin", false),
	ATAN("atan", false),
	AUC("auc", true),
	AUTOENCODER2LAYER("autoencoder_2layer", true),
	AVG_POOL("avg_pool", false),
	AVG_POOL_BACKWARD("avg_pool_backward", false),
	BATCH_NORM2D("batch_norm2d", false, ReturnType.MULTI_RETURN),
	BATCH_NORM2D_BACKWARD("batch_norm2d_backward", false, ReturnType.MULTI_RETURN),
	BIASADD("bias_add", false),
	BIASMULT("bias_multiply", false),
	BANDIT("bandit", true),
	BITWAND("bitwAnd", false),
	BITWOR("bitwOr", false),
	BITWXOR("bitwXor", false),
	BITWSHIFTL("bitwShiftL", false),
	BITWSHIFTR("bitwShiftR", false),
	BIVAR("bivar", true),
	CAST_AS_BOOLEAN("as.logical", "as.boolean", false),
	CAST_AS_DOUBLE("as.double", false),
	CAST_AS_FRAME("as.frame", false),
	CAST_AS_INT("as.integer", false),
	CAST_AS_LIST("as.list", false),
	CAST_AS_MATRIX("as.matrix", false),
	CAST_AS_SCALAR("as.scalar", "castAsScalar", false),
	CBIND("cbind", "append", false),
	CEIL("ceil", "ceiling", false),
	CHOLESKY("cholesky", false),
	COL_COUNT_DISTINCT("colCountDistinct",false),
	COLMAX("colMaxs", false),
	COLMEAN("colMeans", false),
	COLMIN("colMins", false),
	COLNAMES("colnames", false),
	COLPROD("colProds", false),
	COLSD("colSds", false),
	COLSUM("colSums", false),
	COLVAR("colVars", false),
	COMPONENTS("components", true),
	COMPRESS("compress", false, ReturnType.MULTI_RETURN),
	QUANTIZE_COMPRESS("quantize_compress", false, ReturnType.MULTI_RETURN),
	CONFUSIONMATRIX("confusionMatrix", true),
	CONV2D("conv2d", false),
	CONV2D_BACKWARD_FILTER("conv2d_backward_filter", false),
	CONV2D_BACKWARD_DATA("conv2d_backward_data", false),
	COOCCURRENCEMATRIX("cooccurrenceMatrix", true),
	COR("cor", true),
	CORRECTTYPOS("correctTypos", true),
	CORRECTTYPOSAPPLY("correctTyposApply", true),
	COS("cos", false),
	COSH("cosh", false),
	COV("cov", false),
	COX("cox", true),
	CSPLINE("cspline", true),
	CSPLINE_CG("csplineCG", true),
	CSPLINE_DS("csplineDS", true),
	CUMMAX("cummax", false),
	CUMMIN("cummin", false),
	CUMPROD("cumprod", false),
	CUMSUM("cumsum", false),
	CUMSUMPROD("cumsumprod", false),
	DBSCAN("dbscan", true),
	DBSCANAPPLY("dbscanApply", true),
	DECISIONTREE("decisionTree", true),
	DECISIONTREEPREDICT("decisionTreePredict", true),
	DECOMPRESS("decompress", false),
	DEEPWALK("deepWalk", true),
	DET("det", false),
	DETECTSCHEMA("detectSchema", false),
	DENIALCONSTRAINTS("denialConstraints", true),
	DIFFERENCESTATISTICS("differenceStatistics", true),
	DIAG("diag", false),
	DISCOVER_FD("discoverFD", true),
	DISCOVER_MD("mdedup", true),
	SETDIFF("setdiff", true),
	DIST("dist", true),
	DMV("dmv", true),
	DROP_INVALID_TYPE("dropInvalidType", false),
	DROP_INVALID_LENGTH("dropInvalidLength", false),
	EIGEN("eigen", false, ReturnType.MULTI_RETURN),
	EMA("ema", true),
	EXISTS("exists", false),
	EXECUTE_PIPELINE("executePipeline", true),
	EXP("exp", false),
	EVAL("eval", false),
	EVALLIST("evalList", false),
	F1SCORE("f1Score", true),
	FDR("fdr", "FDR", true),
	FIT_PIPELINE("fit_pipeline", true),
	FIX_INVALID_LENGTHS("fixInvalidLengths", true),
	FIX_INVALID_LENGTHS_APPLY("fixInvalidLengthsApply", true),
	FFT("fft", false, ReturnType.MULTI_RETURN),
	FFT_LINEARIZED("fft_linearized", false, ReturnType.MULTI_RETURN),
	FF_TRAIN("ffTrain", true),
	FF_PREDICT("ffPredict", true),
	FLOOR("floor", false),
	FLATTENQUANTILE("flattenQuantile", true),
	FRAME_SORT("frameSort", true),
	FRAME_ROW_REPLICATE("freplicate", false),
	FREQUENCYENCODE("frequencyEncode", true),
	FREQUENCYENCODEAPPLY("frequencyEncodeApply", true),
	GARCH("garch", true),
	GAUSSIAN_CLASSIFIER("gaussianClassifier", true),
	GET_ACCURACY("getAccuracy", true),
	GLM("glm", true),
	GLM_PREDICT("glmPredict", true),
	GLOVE("glove", true),
	GMM("gmm", true),
	GMM_PREDICT("gmmPredict", true),
	GNMF("gnmf", true),
	GRID_SEARCH("gridSearch", true),
	TOPK_CLEANING("topk_cleaning", true),
	HOSPITAL_RESIDENCY_MATCH("hospitalResidencyMatch", true),
	HYPERBAND("hyperband", true),
	IFELSE("ifelse", false),
	IFFT("ifft", false, ReturnType.MULTI_RETURN),
	IFFT_LINEARIZED("ifft_linearized", false, ReturnType.MULTI_RETURN),
	IMG_MIRROR("img_mirror", true),
	IMG_MIRROR_LINEARIZED("img_mirror_linearized", true),
	IMG_BRIGHTNESS("img_brightness", true),
	IMG_BRIGHTNESS_LINEARIZED("img_brightness_linearized", true),
	IMG_CROP("img_crop", true),
	IMG_CROP_LINEARIZED("img_crop_linearized", true),
	IMG_TRANSFORM("img_transform", true),
	IMG_TRANSFORM_LINEARIZED("img_transform_linearized", true),
	IMG_TRANSLATE("img_translate", true),
	IMG_TRANSLATE_LINEARIZED("img_translate_linearized", true),
	IMG_ROTATE("img_rotate", true),
	IMG_ROTATE_LINEARIZED("img_rotate_linearized", true),
	IMG_SHEAR("img_shear", true),
	IMG_SHEAR_LINEARIZED("img_shear_linearized", true),
	IMG_CUTOUT("img_cutout", true),
	IMG_CUTOUT_LINEARIZED("img_cutout_linearized", true),
	IMG_SAMPLE_PAIRING("img_sample_pairing", true),
	IMG_SAMPLE_PAIRING_LINEARIZED("img_sample_pairing_linearized", true),
	IMG_INVERT("img_invert", true),
	IMG_INVERT_LINEARIZED("img_invert_linearized", true),
	IMG_POSTERIZE("img_posterize", true),
	IMG_POSTERIZE_LINEARIZED("img_posterize_linearized", true),
	IMPURITY_MEASURES("impurityMeasures", true),
	IMPUTE_BY_KNN("imputeByKNN", true),
	IMPUTE_BY_MEAN("imputeByMean", true),
	IMPUTE_BY_MEAN_APPLY("imputeByMeanApply", true),
	IMPUTE_BY_MEDIAN("imputeByMedian", true),
	IMPUTE_BY_MEDIAN_APPLY("imputeByMedianApply", true),
	IMPUTE_BY_MODE("imputeByMode", true),
	IMPUTE_BY_MODE_APPLY("imputeByModeApply", true),
	IMPUTE_FD("imputeByFD", true),
	IMPUTE_FD_APPLY("imputeByFDApply", true),
	INCSLICELINE("incSliceLine", true),
	INTERQUANTILE("interQuantile", false),
	INTERSECT("intersect", true),
	INVERSE("inv", "inverse", false),
	IQM("interQuartileMean", false),
	ISNA("is.na", "isNA", false),
	ISNAN("is.nan", "isNaN", false),
	ISINF("is.infinite", "isInf", false),
	KM("km", true),
	KMEANS("kmeans", true),
	KMEANSPREDICT("kmeansPredict", true),
	KNNBF("knnbf", true),
	KNNGRAPH("knnGraph", true),
	KNN("knn", true),
	L2SVM("l2svm", true),
	L2SVMPREDICT("l2svmPredict", true),
	LASSO("lasso", true),
	LENET_TRAIN("lenetTrain", true),
	LENET_PREDICT("lenetPredict", true),
	LENGTH("length", false),
	LINEAGE("lineage", false),
	LIST("list", false),  //note: builtin and parbuiltin
	LM("lm", true),
	LMCG("lmCG", true),
	LMDS("lmDS", true),
	LMPREDICT("lmPredict", true),
	LMPREDICT_STATS("lmPredictStats", true),
	LOCAL("local", false),
	LOG("log", false),
	LOGSUMEXP("logSumExp", true),
	LSTM("lstm", false, ReturnType.MULTI_RETURN),
	LSTM_BACKWARD("lstm_backward", false, ReturnType.MULTI_RETURN),
	LU("lu", false, ReturnType.MULTI_RETURN),
	MAP("map", false),
	MATRIXPROFILE("matrixProfile", true),
	MAX("max", "pmax", false),
	MAX_POOL("max_pool", false),
	MAX_POOL_BACKWARD("max_pool_backward", false),
	MCC("mcc", true),
	MAE("mae", true),
	MAPE("mape", true),
	MEAN("mean", "avg", false),
	MEDIAN("median", false),
	MICE("mice", true),
	MICE_APPLY("miceApply", true),
	MIN("min", "pmin", false),
	MOMENT("moment", "centralMoment", false),
	MSE("mse", true),
	MSMAPE("msmape", true),
	MSVM("msvm", true),
	MSVMPREDICT("msvmPredict", true),
	MULTILOGREG("multiLogReg", true),
	MULTILOGREGPREDICT("multiLogRegPredict", true),
	NA_LOCF("na_locf", true),
	NAIVEBAYES("naiveBayes", true, false),
	NAIVEBAYESPREDICT("naiveBayesPredict", true, false),
	NCOL("ncol", false),
	NORMALIZE("normalize", true),
	NORMALIZEAPPLY("normalizeApply", true),
	NROW("nrow", false),
	NRMSE("nrmse", true),
	OUTER("outer", false),
	OUTLIER("outlier", true, false), //TODO parameterize opposite
	OUTLIER_ARIMA("outlierByArima",true),
	OUTLIER_IQR("outlierByIQR", true),
	OUTLIER_IQR_APPLY("outlierByIQRApply", true),
	OUTLIER_SD("outlierBySd", true),
	OUTLIER_SD_APPLY("outlierBySdApply", true),
	PAGERANK("pageRank", true),
	PCA("pca", true),
	PCAINVERSE("pcaInverse", true),
	PCATRANSFORM("pcaTransform", true),
	PNMF("pnmf", true),
	PPCA("ppca", true),
	PPRED("ppred", false),
	PROD("prod", false),
	PSNR("psnr", true),
	QR("qr", false, ReturnType.MULTI_RETURN),
	QUANTILE("quantile", false),
	QUANTIZEBYCLUSTER("quantizeByCluster", true),
	RANDOM_FOREST("randomForest", true),
	RANDOM_FOREST_PREDICT("randomForestPredict", true),
	RANGE("range", false),
	RAGROUPBY("raGroupby", true),
	RAJOIN("raJoin", true),
	RASELECTION("raSelection", true),
	RBIND("rbind", false),
	RCM("rowClassMeet", "rcm", false, false, ReturnType.MULTI_RETURN),
	REMOVE("remove", false, ReturnType.MULTI_RETURN),
	REV("rev", false),
	ROLL("roll", false),
	ROUND("round", false),
	ROW_COUNT_DISTINCT("rowCountDistinct",false),
	ROWINDEXMAX("rowIndexMax", false),
	ROWINDEXMIN("rowIndexMin", false),
	ROWMAX("rowMaxs", false),
	ROWMEAN("rowMeans", false),
	ROWMIN("rowMins", false),
	ROWPROD("rowProds", false),
	ROWSD("rowSds", false),
	ROWSUM("rowSums", false),
	ROWVAR("rowVars", false),
	RMSE("rmse", true),
	SAMPLE("sample", false),
	SD("sd", false),
	SELVARTHRESH("selectByVarThresh", true),
	SEQ("seq", false),
	SES("ses", true),
	SYMMETRICDIFFERENCE("symmetricDifference", true),
	SHAPEXPLAINER("shapExplainer", true),
	SHERLOCK("sherlock", true),
	SHERLOCKPREDICT("sherlockPredict", true),
	SHORTESTPATH("shortestPath", true),
	SIGMOID("sigmoid", true),   // 1 / (1 + exp(-X))
	SIGN("sign", false),
	SIN("sin", false),
	SINH("sinh", false),
	SLICEFINDER("slicefinder", true), //TODO remove
	SLICELINE("sliceLine", true),
	SLICELINE_DEBUG("sliceLineDebug", true),
	SLICELINE_EXTRACT("sliceLineExtract", true),
	SKEWNESS("skewness", true),
	SMAPE("smape", true),
	SMOTE("smote", true),
	SOFTMAX("softmax", true),
	SOLVE("solve", false),
	SPLIT("split", true),
	SPLIT_BALANCED("splitBalanced", true),
	STABLE_MARRIAGE("stableMarriage", true),
	STATSNA("statsNA", true),
	STRATSTATS("stratstats", true),
	STEPLM("steplm",true, ReturnType.MULTI_RETURN),
	STFT("stft", false, ReturnType.MULTI_RETURN),
	SQRT("sqrt", false),
	SQRT_MATRIX("sqrtMatrix", true),
	SQRT_MATRIX_JAVA("sqrtMatrixJava", false, ReturnType.SINGLE_RETURN),
	SUM("sum", false),
	SVD("svd", false, ReturnType.MULTI_RETURN),
	TABLE("table", "ctable", false),
	TAN("tan", false),
	TANH("tanh", false),
	TO_ONE_HOT("toOneHot", true),
	TOMEKLINK("tomeklink", true),
	TRACE("trace", "tr", false),
	TRANS("t", false),
	TSNE("tSNE", true),
	TYPEOF("typeof", false),
	UNIVAR("univar", true),
	UNION("union", true),
	VAR("var", false),
	VALUE_SWAP("valueSwap", false),
	VECTOR_TO_CSV("vectorToCsv", true),
	WER("wer", true, false),
	WINSORIZE("winsorize", true, false), //TODO parameterize w/ prob, min/max val
	WINSORIZEAPPLY("winsorizeApply", true, false), //TODO parameterize w/ prob, min/max val
	WOE("WoE", true ),
	WOEAPPLY("WoEApply", true ),
	XGBOOST("xgboost", true),
	XGBOOSTPREDICT("xgboostPredict", true),
	XGBOOSTPREDICT_REG("xgboostPredictRegression", true),
	XGBOOSTPREDICT_CLASS("xgboostPredictClassification", true),
	XOR("xor", false),

	// Parameterized functions with parameters
	AUTODIFF("autoDiff", false, true),
	CDF("cdf", false, true),
	CONTAINS("contains", false, true),
	COUNT_DISTINCT("countDistinct",false, true),
	COUNT_DISTINCT_APPROX("countDistinctApprox", false, true),
	COUNT_DISTINCT_APPROX_ROW("rowCountDistinctApprox", false, true),
	COUNT_DISTINCT_APPROX_COL("colCountDistinctApprox", false, true),
	CVLM("cvlm", true, false),
	GROUPEDAGG("aggregate", "groupedAggregate", false, true),
	INVCDF("icdf", false, true),
	LISTNV("list", false, true), //note: builtin and parbuiltin
	LOWER_TRI("lower.tri", false, true),
	ORDER("order", false, true),
	PARAMSERV("paramserv", false, true),
	PCHISQ("pchisq", false, true),
	PEXP("pexp", false, true),
	PF("pf", false, true),
	PNORM("pnorm", false, true),
	PT("pt", false, true),
	QCHISQ("qchisq", false, true),
	QEXP("qexp", false, true),
	QF("qf", false, true),
	QNORM("qnorm", false, true),
	QT("qt", false, true),
	REPLACE("replace", false, true),
	RMEMPTY("removeEmpty", false, true),
	SCALE("scale", true, false),
	SCALEAPPLY("scaleApply", true, false),
	SCALE_MINMAX("scaleMinMax", true, false),
	TIME("time", false),
	TOKENIZE("tokenize", false, true),
	TOSTRING("toString", false, true),
	TRANSFORMAPPLY("transformapply", false, true),
	TRANSFORMCOLMAP("transformcolmap", false, true),
	TRANSFORMDECODE("transformdecode", false, true),
	TRANSFORMENCODE("transformencode", false, true),
	TRANSFORMMETA("transformmeta", false, true),
	UNDER_SAMPLING("underSampling", true),
	UNIQUE("unique", false, true),
	UPPER_TRI("upper.tri", false, true),
	XDUMMY1("xdummy1", true), //error handling test
	XDUMMY2("xdummy2", true), //error handling test
	GETCOLNAMES("getColNames", false, true),
	SETCOLNAMES("setColNames", false, true);

	Builtins(String name, boolean script) {
		this(name, null, script, false, ReturnType.SINGLE_RETURN);
	}

	Builtins(String name, boolean script, ReturnType retType) {
		this(name, null, script, false, retType);
	}

	Builtins(String name, boolean script, boolean parameterized) {
		this(name, null, script, parameterized, ReturnType.SINGLE_RETURN);
	}

	Builtins(String name, String alias, boolean script) {
		this(name, alias, script, false, ReturnType.SINGLE_RETURN);
	}

	Builtins(String name, String alias, boolean script, boolean parameterized) {
		this(name, alias, script, parameterized, ReturnType.SINGLE_RETURN);
	}

	Builtins(String name, String alias, boolean script, boolean parameterized, ReturnType retType) {
		_name = name;
		_alias = alias;
		_script = script;
		_parameterized = parameterized;
		_retType = retType;
	}

	private final static String BUILTIN_DIR = "scripts/builtin/";
	private final static HashMap<String, Builtins> _map = new HashMap<>();

	static {
		//materialize lookup map for all builtin names
		for( Builtins b : EnumSet.allOf(Builtins.class) ) {
			_map.put(b.getName(), b);
			if( b.getAlias() != null )
				_map.put(b.getAlias(), b);
		}
	}

	private final String _name;
	private final String _alias;
	private final boolean _script;
	private final boolean _parameterized;
	private final ReturnType _retType;

	public String getName() {
		return _name;
	}

	public String getAlias() {
		return _alias;
	}

	public boolean isScript() {
		return _script;
	}

	public boolean isParameterized() {
		return _parameterized;
	}

	public boolean isMultiReturn() {
		return _retType == ReturnType.MULTI_RETURN;
	}

	public static boolean contains(String name, boolean script, boolean parameterized) {
		Builtins tmp = get(name);
		return tmp != null && script == tmp.isScript()
				&& parameterized == tmp.isParameterized();
	}

	public static Builtins get(String name) {
		if( name.equals("list") )
			return LIST; //unparameterized
		return _map.get(name);
	}

	public static Builtins get(String name, boolean params) {
		if( name.equals("list") )
			return params ? LISTNV : LIST;
		Builtins tmp = get(name);
		return tmp != null && (params == tmp.isParameterized()) ? tmp : null;
	}

	public static String getFilePath(String name) {
		StringBuilder sb = new StringBuilder();
		sb.append(BUILTIN_DIR);
		sb.append(name);
		sb.append(".dml");
		return sb.toString();
	}

	public static String getInternalFName(String name, DataType dt) {
		return !contains(name, true, false) ? name : // private builtin
				(dt.isMatrix() ? "m_" :              // public builtin
				 dt.isFrame() ? "f_" : "s_") + name;
	}
}
