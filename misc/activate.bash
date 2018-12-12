thisdir="$(dirname "${BASH_SOURCE[0]}")"
source "$thisdir/rc/rc.sh"

# Use readlink instead of realpath since it is available in more
# environments:
env="${TC_GAN_ENV:-$(readlink --canonicalize $thisdir/../env)}"

export PATH="$env/bin:$PATH"
export CPATH="$env/include:$CPATH"
export LIBRARY_PATH="$env/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$env/lib:$LD_LIBRARY_PATH"

# In principle, this is better done via env/bin/activate like this:
#
#     source "$thisdir/../env/bin/activate" "$thisdir/../env"
#
# However, as conda tries to symlink stuff during activation, it's not
# a good idea to use it in a script to be called in parallel (see,
# e.g., https://github.com/conda/conda/issues/3001).

# Just following the instruction (via RuntimeError) from Theano.
# https://github.com/Theano/Theano/issues/6499#issuecomment-342160463
export MKL_THREADING_LAYER=GNU
