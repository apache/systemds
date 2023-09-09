package org.apache.sysds.utils;

public class ImgNativeHelper extends NativeHelper {

    //image rotation
    public static native void imageRotate(double[] img_in, int rows, int cols, double radians, double fill_value, double[] img_out);

    //image cutout
    public static native double[] imageCutout(double[] img_in, int rows, int cols, int x, int y, int width, int height, double fill_value);

    //image crop
    public static native double[] cropImage(double[] img_in, int orig_w, int orig_h, int w, int h, int x_offset, int y_offset);

    //image translate
    public static native void imgTranslate(double[] img_in, double offset_x, double offset_y,
                                           int in_w, int in_h, int out_w, int out_h,
                                           double fill_value, double[] img_out);
}
