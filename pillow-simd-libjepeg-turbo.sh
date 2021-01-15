conda uninstall -y --force pillow pil pillow-simd jpeg libtiff libjpeg-turbo
pip   uninstall -y         pillow pil pillow-simd jpeg libtiff libjpeg-turbo
conda install -yc conda-forge libjpeg-turbo
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd==7.0.0.post3
conda install -y jpeg libtiff