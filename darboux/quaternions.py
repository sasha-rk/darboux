import numpy as np

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