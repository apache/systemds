package org.apache.sysds.runtime.io.cog;

/**
 * Enum for mapping IFD tag values to their corresponding tag names
 */
public enum IFDTagDictionary {
    Unknown(-1),
    // Right now we will only support baseline TIFF
    // not the extended version
    NewSubfileType(254),
    ImageWidth(256),
    ImageLength(257),
    BitsPerSample(258),
    Compression(259),
    PhotometricInterpretation(262),
    Threshholding(263),
    CellWidth(264),
    CellLength(265),
    FillOrder(266),
    ImageDescription(270),
    Make(271),
    Model(272),
    StripOffsets(273),
    Orientation(274),
    SamplesPerPixel(277),
    RowsPerStrip(278),
    StripByteCounts(279),
    MinSampleValue(280),
    MaxSampleValue(281),
    XResolution(282),
    YResolution(283),
    PlanarConfiguration(284),
    FreeOffsets(288),
    FreeByteCounts(289),
    GrayResponseUnit(290),
    GrayResponseCurve(291),
    ResolutionUnit(296),
    Software(305),
    DateTime(306),
    Artist(315),
    HostComputer(316),
    ColorMap(320),
    ExtraSamples(338),
    /**
     * 1 = unsigned integer data
     * 2 = two's complement signed integer data
     * 3 = IEEE floating point data [IEEE]
     * 4 = undefined data format
     * Has as many values as SamplesPerPixel
     */
    SampleFormat(339),

    // Extended tags we need (COG specifically)
    TileWidth(322),
    TileLength(323),
    TileOffsets(324),
    TileByteCounts(325),
    GDALNoData(42113),
    GeoKeyDirectoryTag(34735),
    GeoDoubleParamsTag(34736),
    GeoAsciiParamsTag(34737),
    ModelPixelScaleTag(33550),
    ModelTiepointTag(33922),
    ModelTransformationTag(34264);


    private final int value;

    IFDTagDictionary(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public static IFDTagDictionary valueOf(int value) {
        for (IFDTagDictionary tag : IFDTagDictionary.values()) {
            if (tag.getValue() == value) {
                return tag;
            }
        }
        return null;
    }
}
