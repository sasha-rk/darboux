import ddg
import numpy as np
import bpy

# --- Dual quaternion helpers ---
def q_conj(q):  # quaternion conjugate
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(a, b):  # quaternion product
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    return np.array([
        a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4,
        a1 * b2 + a2 * b1 - a4 * b3 + a3 * b4,
        a1 * b3 + a3 * b1 + a4 * b2 - a2 * b4,
        a1 * b4 + a4 * b1 - a3 * b2 + a2 * b3
    ])

def apply_se3(r, t, x = np.array([1, 2, 3])):
    """
    applying a rotation rx\bar{r} and a translation t to the given point
    """
    x_embed = np.insert(x, 0, 0)
    r = np.asarray(r).reshape(-1)
    t = np.asarray(t).reshape(-1)[1:]
    x_rot = q_mul(r, q_mul(x_embed, q_conj(r)))[1:]
    return x_rot + t

def se3_from_study(q):
    """
    q is from Study quadric, so it is given by a 7 dim vector in homogeneous coordinates.
    """
    qh = ddg.math.projective.homogenize(q)
    q_0 = q[:4]
    norm_r = np.linalg.norm(q_0)

    # getting rid of unwanted points from the exceptional plane 
    if norm_r < 1e-8:
        return None, None
    
    qh = qh / norm_r 
    return qh[:4], 2 * q_mul(qh[4:], q_conj(qh[:4]))

def motion_from_se3(r, q, x = np.array([0, 0, 0, 0])):
    return

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

# E = {r + eps*q | r = 0} the exceptional 3-plane
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

L = np.eye(4)
M = 0.12 * np.diag([1.0, -7, 5, -3.0])  # small norm → far from E & compact motion
A_matrix = np.vstack([L, M])                # 8x4
A = ddg.geometry.subspace_from_columns(A_matrix)

# intersection of S with A. which should be a quadric in the space spanned by vectors of A
I = ddg.geometry.intersect(S, A)

# --- your intersection I is already built ---
N = ddg.to_smooth_net(I)  # I is a quadric surface in a 3-space inside RP^7
D = ddg.nets.sample_smooth_net(N, sampling=[[0.05, 200, 'c'], [0.05, 200, 'c']])

# pick a seed point to move (trajectory surface of this point)
p0 = np.array([0, 0, 0])

def study_act_from_affine(h):
    r, t = se3_from_study(h)
    if r is None:
        # CHG6: return finite 3-vector even for invalid poses to avoid KeyError in conversion
        return np.array([np.nan, np.nan, np.nan])
    return apply_se3(r, t, p0)

# CHG7: compose the kinematic map with the DISCRETE surface
# Ssurf = ddg.nets.compose(study_act_from_affine, D)

# now embed & act just like before
Ssurf = ddg.nets.compose(study_act_from_affine, D)

red = ddg.blender.material.material(color=(1.,0.,0.), specular=0.5, roughness=0.1, alpha=1.0)

# --- Scene setup ---
ddg.blender.scene.clear(deep=True)
light = ddg.blender.light.light("sun", location=(20,20,20))
# - normal cam - 
cam = ddg.blender.camera.camera(location=(8, 8, 8))
ddg.blender.camera.look_at_point(cam, (0, 0, 0))
bpy.context.scene.camera = cam

ddg.blender.convert(Ssurf, "motion surface", red)