"""Make the tests import the SOURCE tree, wherever pytest is invoked from.

Without this, running from the workspace root (`pytest src/object_tracker/tests`)
resolves `kf_object_tracking` to the colcon-installed copy under
`install/kf_object_tracking/...` instead of the source next to these tests. That
copy is only as fresh as the last `colcon build`, so the suite silently grades a
stale build artifact — it passed from one directory and failed from another,
with no indication why.

Unit tests should exercise the code as written, so the package root goes on the
front of sys.path.
"""

import os
import sys

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if sys.path[:1] != [PKG_ROOT]:
    while PKG_ROOT in sys.path:
        sys.path.remove(PKG_ROOT)
    sys.path.insert(0, PKG_ROOT)
