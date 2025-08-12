import numpy as np
import ddg
import bpy

# --- Materials ---
white = ddg.blender.material.material(color=(1.,1.,1.), specular=0.5, roughness=0.1, alpha=1.0)
red = ddg.blender.material.material(color=(1.,0.,0.), specular=0.5, roughness=0.1, alpha=1.0)

# --- Scene setup ---
ddg.blender.scene.clear(deep=True)
light = ddg.blender.light.light("sun", location=(20,20,20))
# setup_area_lights(shadow=True)
# - normal cam - 
cam = ddg.blender.camera.camera(location=(8, 8, 8))
ddg.blender.camera.look_at_point(cam, (0, 0, 0))
bpy.context.scene.camera = cam

collection = ddg.blender.collection.collection("twist")

# 1) A quadric (example: one-sheet hyperboloid)
Q = ddg.geometry.Quadric(np.diag([1, 1, -1, -1]))

# 2) Turn the quadric into a smooth parametric net (or NetCollection if multi-sheet)
N = ddg.to_smooth_net(Q)  # supports quadrics; returns SmoothNet/NetCollection

# 3) Define any transformation φ: R^3 → R^3  (here: a twist around z)
def twist(p):
    x, y, z = p
    th = 0.8*z
    c, s = np.cos(th), np.sin(th)
    return np.array([c*x - s*y, s*x + c*y, z])

# Compose φ ∘ N (works for nets and net collections)
Ndef = ddg.nets.compose(twist, N)

# 4) Sample the smooth net to a discrete net (needed for Blender conversion)
# sampling format: [step_u, num_u, 'c'] per direction (see docs examples)
D = ddg.nets.sample_smooth_net(Ndef, [0.1, 50, 'c'])  # surface: sampler uses both directions

# 5) Convert to Blender (a mesh object)
bobj = ddg.blender.convert(D, "Twisted hyperboloid", red, collection, (5, 5, 5))