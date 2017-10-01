thisdir="$(dirname "${BASH_SOURCE[0]}")"
source "$thisdir/rc/rc.sh"

PATH="$thisdir/../env/bin:$PATH"
# In principle, this is better done via env/bin/activate like this:
#
#     source "$thisdir/../env/bin/activate" "$thisdir/../env"
#
# However, as conda tries to symlink stuff during activation, it's not
# a good idea to use it in a script to be called in parallel (see,
# e.g., https://github.com/conda/conda/issues/3001).
