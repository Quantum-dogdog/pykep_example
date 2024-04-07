# pykep_example

因为网上的pykep算法过时了，所以上传了两个使用pykep的算例，一个是mga_1dsm，一个是mga_lt_nep，适用于python==3.7.12，pykep==2.6，pygmo==2.18.0版本。
但是对于具体的算法，比如nlopt中的cobyla算法是什么，并没有多少了解，只是这样写能够work，只是这个算法可以处理nonlinear constrain，而ipopt不行（不会设置gradient），nega2不行（不能解这种类型的问题），供大家参考。
pykep是用来设置问题的参数，pygmo是用来设置求解的算法。

# environment setting
为了避免冲突，需要先安装pygmo，再安装pykep.

# packages in environment at e:\mambaforge\envs\esaEnv:
#
# Name                    Version                   Build  Channel
backcall                  0.2.0              pyh9f0ad1d_0    conda-forge
backports                 1.0                pyhd8ed1ab_3    conda-forge
backports.functools_lru_cache 2.0.0              pyhd8ed1ab_0    conda-forge
boost                     1.74.0           py37h457096e_5    conda-forge
boost-cpp                 1.74.0               h9f4b32c_8    conda-forge
bzip2                     1.0.8                hcfcfb64_5    conda-forge
ca-certificates           2024.2.2             h56e8100_0    conda-forge
certifi                   2024.2.2           pyhd8ed1ab_0    conda-forge
cloudpickle               2.2.1              pyhd8ed1ab_0    conda-forge
colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
cycler                    0.11.0             pyhd8ed1ab_0    conda-forge
debugpy                   1.6.3            py37hf2a7229_0    conda-forge
decorator                 4.4.2                      py_0    conda-forge
entrypoints               0.4                pyhd8ed1ab_0    conda-forge
freetype                  2.12.1               hdaf720e_2    conda-forge
importlib-metadata        6.7.0                    pypi_0    pypi
intel-openmp              2024.0.0         h57928b3_49841    conda-forge
ipopt                     3.14.9               h0ff4939_1    conda-forge
ipykernel                 6.16.2             pyh025b116_0    conda-forge
ipyparallel               8.6.1              pyhd8ed1ab_0    conda-forge
ipython                   7.33.0           py37h03978a9_0    conda-forge
jedi                      0.19.1             pyhd8ed1ab_0    conda-forge
joblib                    1.3.2                    pypi_0    pypi
jpeg                      9e                   hcfcfb64_3    conda-forge
jupyter_client            7.4.9              pyhd8ed1ab_0    conda-forge
jupyter_core              4.11.1           py37h03978a9_0    conda-forge
kiwisolver                1.3.2            py37h8c56517_1    conda-forge
lcms2                     2.14                 h90d422f_0    conda-forge
lerc                      4.0.0                h63175ca_0    conda-forge
libblas                   3.9.0              21_win64_mkl    conda-forge
libcblas                  3.9.0              21_win64_mkl    conda-forge
libdeflate                1.14                 hcfcfb64_0    conda-forge
libflang                  5.0.0           h6538335_20180525    conda-forge
libhwloc                  2.9.3           default_haede6df_1009    conda-forge
libiconv                  1.17                 hcfcfb64_2    conda-forge
liblapack                 3.9.0              21_win64_mkl    conda-forge
libpng                    1.6.43               h19919ed_0    conda-forge
libsodium                 1.0.18               h8d14728_1    conda-forge
libsqlite                 3.45.2               hcfcfb64_0    conda-forge
libtiff                   4.4.0                hc4f729c_5    conda-forge
libwebp-base              1.3.2                hcfcfb64_0    conda-forge
libxcb                    1.13              hcd874cb_1004    conda-forge
libxml2                   2.12.6               hc3477c8_1    conda-forge
libzlib                   1.2.13               hcfcfb64_5    conda-forge
llvm-meta                 5.0.0                         0    conda-forge
llvmlite                  0.39.1                   pypi_0    pypi
loguru                    0.7.2                    pypi_0    pypi
m2w64-gcc-libgfortran     5.3.0                         6    conda-forge
m2w64-gcc-libs            5.3.0                         7    conda-forge
m2w64-gcc-libs-core       5.3.0                         7    conda-forge
m2w64-gmp                 6.1.0                         2    conda-forge
m2w64-libwinpthread-git   5.0.0.4634.697f757               2    conda-forge
matplotlib-base           3.4.3            py37h4a79c79_2    conda-forge
matplotlib-inline         0.1.6              pyhd8ed1ab_0    conda-forge
metis                     5.1.0             h63175ca_1007    conda-forge
mkl                       2024.0.0         h66d3029_49657    conda-forge
msys2-conda-epoch         20160418                      1    conda-forge
mumps-seq                 5.2.1               h1f49738_14    conda-forge
nest-asyncio              1.6.0              pyhd8ed1ab_0    conda-forge
networkx                  2.5.1              pyhd8ed1ab_0    conda-forge
nlopt                     2.7.1            py37h1d391ab_1    conda-forge
numba                     0.56.4                   pypi_0    pypi
numpy                     1.21.6           py37h2830a78_0    conda-forge
openjpeg                  2.5.0                hc9384bd_1    conda-forge
openmp                    5.0.0                    vc14_1    conda-forge
openssl                   3.2.1                hcfcfb64_1    conda-forge
packaging                 23.2               pyhd8ed1ab_0    conda-forge
pagmo                     2.18.0               h1c250d2_4    conda-forge
parso                     0.8.3              pyhd8ed1ab_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.2.0            py37h42a8222_2    conda-forge
pip                       24.0               pyhd8ed1ab_0    conda-forge
prompt-toolkit            3.0.42             pyha770c72_0    conda-forge
psutil                    5.9.3            py37h51bd9d9_0    conda-forge
pthread-stubs             0.4               hcd874cb_1001    conda-forge
pthreads-win32            2.9.1                hfa6e2cd_3    conda-forge
pybind11-abi              4                    hd8ed1ab_3    conda-forge
pygments                  2.17.2             pyhd8ed1ab_0    conda-forge
pygmo                     2.18.0           py37h8d76dce_2    conda-forge
pykep                     2.6              py37h959d761_5    conda-forge
pyparsing                 3.1.2              pyhd8ed1ab_0    conda-forge
python                    3.7.12          h900ac77_100_cpython    conda-forge
python-dateutil           2.9.0              pyhd8ed1ab_0    conda-forge
python_abi                3.7                     4_cp37m    conda-forge
pywin32                   303              py37hcc03f2d_0    conda-forge
pyzmq                     24.0.1           py37h7347f05_0    conda-forge
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3            py37hb6553fb_0    conda-forge
setuptools                65.6.3             pyhd8ed1ab_0    conda-forge
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.45.2               hcfcfb64_0    conda-forge
tbb                       2021.11.0            h91493d7_1    conda-forge
threadpoolctl             3.1.0                    pypi_0    pypi
tk                        8.6.13               h5226925_1    conda-forge
tornado                   6.2              py37hcc03f2d_0    conda-forge
tqdm                      4.66.2             pyhd8ed1ab_0    conda-forge
traitlets                 5.9.0              pyhd8ed1ab_0    conda-forge
typing-extensions         4.7.1                    pypi_0    pypi
ucrt                      10.0.22621.0         h57928b3_0    conda-forge
vc                        14.3                hcf57466_18    conda-forge
vc14_runtime              14.38.33130         h82b7239_18    conda-forge
vs2015_runtime            14.38.33130         hcb4865c_18    conda-forge
wcwidth                   0.2.10             pyhd8ed1ab_0    conda-forge
wheel                     0.34.2                   py37_0    conda-forge
win32-setctime            1.1.0                    pypi_0    pypi
xorg-libxau               1.0.11               hcd874cb_0    conda-forge
xorg-libxdmcp             1.1.3                hcd874cb_0    conda-forge
xz                        5.2.6                h8d14728_0    conda-forge
zeromq                    4.3.4                h0e60522_1    conda-forge
zipp                      3.15.0                   pypi_0    pypi
zstd                      1.5.5                h12be248_0    conda-forge



