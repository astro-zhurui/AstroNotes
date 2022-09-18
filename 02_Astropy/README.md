# Astropy学习笔记

[Astropy](https://docs.astropy.org/en/stable/index.html#)天体物理领域的核心python包，功能十分强大。我将按[官方文档](https://docs.astropy.org/en/stable/index.html#user-documentation)的结构顺序，按需学习，笔记为jupyter notebook形式。

## Outline

### 1_data-structures (astropy基本数据结构)
    1. `1_physical-quantities` (astropy.constants & astropy.units)
    2. `2_n-dimensinal-datasets` (astropy.nddata)
    3. `3_table` (astropy.table)
    4. `4_coordinates` (astropy.coordinates & astropy.wcs)
    5. `5_time` (astropy.time & astropy.timeseries)

### 2_computation (进行一些计算)
    1. `1_statistics` (astropy.stats)
    2. `2_uncertainties` (astropy.uncertainty)
    3. `3_cosmology` (astropy.cosmology)
    4. `4_convolution` (astropy.convolution)
    5. `5_model-fitting` (astropy.modeling)

### 3_IO (文件的读写)
    1. `1_fits-read` (介绍fits文件读取，以及FITS文件的header、image、table的相关操作；astropy.io.fits)
    2. `2_fits-save` (介绍fits文件的保存，相关HDU、HDUList的创建；astropy.io.fits)
    3. `2_ascii` (astropy.io.ascii)
    4. `3_votable` (astropy.io.votable)

### 4_visualization (数据可视化; astropy.visualization)
    1. `1_plot-quantities.ipynb` (直接在图上画出astropy物理量)
    2. `2_plot-image-with-WCS-coordinates.ipynb` (画带坐标的图像)
    3. `3_image-stretch-and-norm.ipynb` (图像的拉伸与归一化)

### 5_utilities (实用功能)
    1. `1_image-cutout.ipynb`
    2. `2_catalog-matching.ipynb`