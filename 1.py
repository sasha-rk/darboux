import ddg
import numpy as np
import bpy

# study quadric is given bas a set of points in homogeneous coordinates
# {(q0, q1, q2, q3, r0, r1, r2, r3) | q0r0 + q1r1 + q2r2 + q3r3 = 0}
# here is the Gram matrix of the corresponding quadratic form
study_matrix = np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0]])

# S is the Study quadric itself
S = ddg.geometry.Quadric(study_matrix)

# let's also define the exceptional 3-plane
# E = {r + eps*q | r = 0}
E_matrix = np.array([[0, 0, 0, 0,],
                     [0, 0, 0, 0,],
                     [0, 0, 0, 0,],
                     [0, 0, 0, 0,],
                     [1, 0, 0, 0,],
                     [0, 1, 0, 0,],
                     [0, 0, 1, 0,],
                     [0, 0, 0, 1,],
                     ])
E = ddg.geometry.subspace_from_columns(E_matrix)

# now we want some 3d subspace of RP7 A spanned by 4 vectors
A_matrix1 = np.array([[1, 0, 0, 0,],
                     [0, 1, 0, 0,],
                     [0, 0, 1, 0,],
                     [0, 0, 0, 0,],
                     [1, 0, 0, 1,],
                     [0, 1, 0, 0,],
                     [0, 0, 1, 0,],
                     [0, 0, 0, 1,],
                     ])
L = np.array([[1.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.2, 0.0, 0.0, 1.0]])     # invertible → A∩E = ∅

M = np.array([[0.0,  0.2, 0.0,  0.0],
              [0.2,  0.0, 0.0,  0.0],
              [0.0,  0.0,-0.6,  0.15],
              [0.0,  0.0, 0.15, 0.4]])   # symmetric, indefinite
A_matrix2 = np.vstack([L, M])

L = np.eye(4)

M = np.diag([0.8, -0.5, 0.3, -0.2])      # diagonal, indefinite → closed/open cones in chart

A_matrix = np.vstack([L, M])
A = ddg.geometry.subspace_from_columns(A_matrix)

# intersection of S with A. which should be a quadric in the space spanned by vectors of A
I = ddg.geometry.intersect(S, A)

# --- your intersection I is already built ---
N = ddg.to_smooth_net(I)  # I is a quadric surface in a 3-space inside RP^7
D = ddg.nets.sample_smooth_net(N, sampling=[[0.05, 160, 'c'], [0.05, 160, 'c']])

# --- Dual quaternion helpers ---
def q_conj(q):  # quaternion conjugate
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(a, b):  # quaternion product
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def rotate_point_by_q(p, q):  # v' = q v q̄ with v=(0,p)
    v = np.array([0.0, p[0], p[1], p[2]])
    return q_mul(q_mul(q, v), q_conj(q))[1:]

def se3_from_study(x8):
    q = x8[:4]; r = x8[4:]
    n = np.linalg.norm(q)
    if n < 1e-8:            # exceptional plane: q≈0 → not a displacement
        return None, None
    qn = q / n              # homogeneous scale → unit quaternion
    t = 2.0 * q_mul(r, q_conj(qn))[1:]  # translation from dual part
    return qn, t

# pick a seed point to move (trajectory surface of this point)
p0 = np.array([1.5, 1.0, 0.3])

def study_act_from_affine(xyz):
    xyz = np.asarray(xyz).reshape(-1)
    x4 = np.array([xyz[0], xyz[1], xyz[2], 1.0])  # affine → homogeneous in RP^3
    X8 = A_matrix @ x4                             # embed into RP^7 basis (q,r)
    q, t = se3_from_study(X8)
    if q is None:
        # CHG6: return finite 3-vector even for invalid poses to avoid KeyError in conversion
        return np.array([np.nan, np.nan, np.nan])
    return rotate_point_by_q(p0, q) + t

# CHG7: compose the kinematic map with the DISCRETE surface
# Ssurf = ddg.nets.compose(study_act_from_affine, D)

# resolution (angles per direction)
nu, nv = 180, 160

# periodic domain → affine chart of A (pick a 2π-periodic map g)
def g(theta, phi):
    # example: wrap two angles through a stereographic torus map into your chart
    # (you can tune this to your actual quadric parameters)
    a, b = 0.85, 0.6
    x = (a + b*np.cos(theta)) * np.cos(phi)
    y = (a + b*np.cos(theta)) * np.sin(phi)
    z = b*np.sin(theta)
    return np.array([x, y, z])

G = ddg.nets.SmoothNet(g, [[2*np.pi/nu, nu, 'c'],
                           [2*np.pi/nv, nv, 'c']])  # (θ,φ) ↦ (x,y,z) in the A-chart
# sample on S^1×S^1 (wrap in both directions)
D_aff = ddg.nets.sample_smooth_net(G, [[2*np.pi/nu, nu, 'c'],
                                       [2*np.pi/nv, nv, 'c']])

# now embed & act just like before
Ssurf = ddg.nets.compose(study_act_from_affine, D_aff)

red = ddg.blender.material.material(color=(1.,0.,0.), specular=0.5, roughness=0.1, alpha=1.0)

# --- Scene setup ---
ddg.blender.scene.clear(deep=True)
light = ddg.blender.light.light("sun", location=(20,20,20))
# setup_area_lights(shadow=True)
# - normal cam - 
cam = ddg.blender.camera.camera(location=(8, 8, 8))
ddg.blender.camera.look_at_point(cam, (0, 0, 0))
bpy.context.scene.camera = cam

ddg.blender.convert(Ssurf, "motion surface", red)