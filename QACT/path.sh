# NOTE: Adjust MAIN_ROOT and KALDI_ROOT to match your ESPnet installation path.
# Example: if QACT/ is placed at /path/to/espnet/egs/swbd/asr1/
#   then MAIN_ROOT should be /path/to/espnet
PA=$(cd "$(dirname "$0")" && pwd)
MAIN_ROOT=${MAIN_ROOT:-$PA/../../espnet/}
KALDI_ROOT=${KALDI_ROOT:-$MAIN_ROOT/tools/kaldi}

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8

# Add QACT model directory to PYTHONPATH so that `model.*` can be imported
export PYTHONPATH=$PA:$PYTHONPATH
