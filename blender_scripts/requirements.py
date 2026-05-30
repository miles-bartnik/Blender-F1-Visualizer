# F1 Blender Import Script — requirements
# These packages must be installed into Blender's bundled Python.
#
# Blender ships its own Python interpreter. To install packages into it:
#
#   Windows:
#     "<blender_install>\<version>\python\bin\python.exe" -m pip install -r requirements_blender.txt
#
#   macOS/Linux:
#     <blender_install>/<version>/python/bin/python3.x -m pip install -r requirements_blender.txt
#
# Note: bpy (Blender Python API) is provided by Blender itself and
# should not be installed via pip.
#
# Python >= 3.11 recommended (bundled with Blender 4.x/5.x)
 
# Cubic spline resampling (track and pit lane curve smoothing)
scipy>=1.11
 
# Numerical computation
numpy>=1.26
 
# Sea polygon vertex group assignment
shapely>=2.0
 