from enum import Enum


class FitsKeys(Enum):
    BG_EXTR = "Application used for Background Extraction"
    GXSTRTCH = "Applied stretch value, Application GraXpert"
    GXINTOPT = "BGE interpolation type, GraXpert"
    GXSMOOTH = "BGE smoothing value, GraXpert"
    GXCORRT = "BGE correction type, GraXpert"
    GXBGAIV = "BGE ai version, GraXpert"
    GXSAMPSZ = "Sample points size, GraXpert"
    GXRBFK = "RBF kernel type, GraXpert"
    GXSPLORD = "BGE spline order, GraXpert"
    GXBGPTS = "Sample points coordinates, GraXpert"
    GXP = "Single sample point coordinates (XISF), GraXpert"  # key prefix, followed by a 5-digit index
    GXSFSCAL = "Sample-free BGE scale, GraXpert"
    GXSFSMTH = "Sample-free BGE smoothness, GraXpert"
    GXSFPROT = "Sample-free BGE protection enabled, GraXpert"
    GXSFTHR = "Sample-free BGE protection threshold, GraXpert"
    GXSFAMT = "Sample-free BGE protection amount, GraXpert"
    GXSFSIMP = "Sample-free BGE simplified mode, GraXpert"
    GXSFDEG = "Sample-free BGE polynomial degree, GraXpert"
    GXSFDOWN = "Sample-free BGE downsample factor, GraXpert"
