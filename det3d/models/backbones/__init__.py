import importlib
spconv_spec = importlib.util.find_spec("spconv")
found = spconv_spec is not None

if found:
    from .scn import SpMiddleResNetFHD, SpMiddleResNetVoxel, VoxelBackBone8x, VoxelBackBoneEnc8x
    from .pcn import *
    from .dcn import *
    from .ppcn import *
else:
    print("No spconv, sparse convolution disabled!")

