import ddg
import numpy as np
import bpy

def make_A_no_E(eps=0.55, seed=228):
    """
    returns an 8x4 matrix A = [L; eps*M] with L invertible ⇒ A ∩ E = ∅.
    eps small for better numerics
    """
    rng = np.random.default_rng(seed)

    # well-conditioned L
    Q, _ = np.linalg.qr(rng.standard_normal((4,4)))
    S = np.diag(1.5 - 0.3*np.arange(4))   # spread singular ues a bit
    L = Q @ S                              

    M = rng.standard_normal((4,4))
    A = np.vstack([L, eps*M])
    return A

# --- Quaternion helpers ---
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

def apply_se3(r, t, x = np.array([1, 0, 0])):
    """
    x 3dim vector. r,t quaternions representing rotation and translation
    return action of (r,t) on x equal r x \bar r + t. cut the first coordinate out
    """
    x_embed = np.insert(x, 0, 0)
    r = np.asarray(r).reshape(-1)
    t = np.asarray(t).reshape(-1)[1:]
    x_rot = q_mul(r, q_mul(x_embed, q_conj(r)))[1:]
    return x_rot + t

def se3_from_study(q):
    """
    takes q an element of the study quadric. 
    homogenize if it's given in affine coordinates
    (q_0, q_1) in S represents r + 0.5 eps t r
    extract motion quaternions r and t with normalized r
    return r, t
    """
    if q.size == 7:
        qh = ddg.math.projective.homogenize(q)
    else:
        qh = q
    norm_r = np.linalg.norm(qh[:4])

    # getting rid of unwanted points from the exceptional plane 
    if norm_r < 1e-16:
        return None, None
    
    qh = qh / norm_r 
    return qh[:4], 2 * q_mul(qh[4:], q_conj(qh[:4]))

def study_act_on_p(p):
    """
    given a starting point p
    return a function which calculates the orbit of p under the action of given h in S
    """
    def f(h):
        r, t = se3_from_study(h)
        if r is None:
            return np.array([np.nan, np.nan, np.nan])
        return apply_se3(r, t, p)
    return f

# study quadric is given as a set of points in homogeneous coordinates
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

def get_set(a):
    """
    generate a random 3 plane in rp7 with seed parameter a
    intersect it with S
    give the intersection, which is a quadric as a parametrized surface back
    """
    A = ddg.geometry.subspace_from_columns(make_A_no_E(seed = a))

    # intersection of S with A. which should be a quadric in the space spanned by vectors of A
    I = ddg.geometry.intersect(S, A)

    # --- your intersection I is already built ---
    N = ddg.to_smooth_net(I)  # I is a quadric surface in a 3-space inside RP^7
    return N

# pick a starting point to move (trajectory surface of this point)
p0 = np.array([1, 0, 0])

# colors
red = ddg.blender.material.material(color=(1.,0.,0.), specular=0.7, roughness=0.3, alpha=01.0)
red_transparent = ddg.blender.material.material(color=(1.,0.,0.), specular=0.7, roughness=0.3, alpha=0.57)
green = ddg.blender.material.material(color=(0.,1.,0.), specular=0.7, roughness=0.3, alpha=1.0)
blue = ddg.blender.material.material(color=(0.,0.,1.), specular=0.7, roughness=0.3, alpha=1.0)
yellow = ddg.blender.material.material(color=(0.,1.,1.), specular=0.7, roughness=0.3, alpha=1.0)
purple = ddg.blender.material.material(color=(1.,0.,1.), specular=0.7, roughness=0.3, alpha=1.0)
black = ddg.blender.material.material(color=(0.,0.,0.), specular=0.7, roughness=0.3, alpha=1.0)

def setup_area_lights(
    point=(1.0, 0.0, 0.0),
    light_type="AREA",
    size=1.0,
    distance=5.0,
    energy=250.0,
    specular=0.0,
    shadow=True,
):
    """
    Set up four lights: left, right, front, and back, that look at
    a given point from a given distance.

    Parameters
    ----------
    point : Iterable of three floats (default=(0., 0., 0.))
        Point that the area lights look at.
    light_type : str (default='AREA')
        Type of the light, 'POINT', 'SUN', 'SPOT' or 'AREA'
    distance: float (default=2.0)
       Distance of area lights from the given point.
    size: float (default=1.0)
        Size of the area lights.
    energy: float (default=50.0)
        Energy of the light in Watts.
    specular : float (default=0.0)
        Specular value of the light in the illumination model.
    shadow: bool (default=False)
        Enable or Disable shadows.

    Returns
    -------
    Iterable of four bpy.types.Light
    """
    top_light = ddg.blender.light.light(
        name="Top Light",
        type_=light_type,
        location=(0.0, 0.0, 2 * distance),
        energy=energy,
        specular=specular,
    )
    top_light.rotation_euler = (0.0, 0.0, np.pi / 2)

    back_light = ddg.blender.light.light(
        name="Back Light",
        type_=light_type,
        location=(0.0, 2 * distance, distance),
        energy=energy,
        specular=specular,
    )
    back_light.rotation_euler = (np.deg2rad(45), 0.0, np.pi)

    right_light = ddg.blender.light.light(
        name="Right Light",
        type_=light_type,
        location=(2 * distance, -2 * distance, distance),
        energy=energy,
        specular=specular,
    )
    right_light.rotation_euler = (np.deg2rad(55), 0.0, -np.deg2rad(45))

    left_light = ddg.blender.light.light(
        name="Left Light",
        type_=light_type,
        location=(-2 * distance, -2 * distance, distance),
        energy=energy,
        specular=specular,
    )
    left_light.rotation_euler = (np.deg2rad(55), 0.0, np.deg2rad(45))

    for light_object in [top_light, back_light, right_light, left_light]:
        if light_object.type == "AREA":
            light_object.data.size = size
        light_object.data.use_shadow = shadow
        ddg.blender.light.look_at_point(light_object, point=point)

    return top_light, back_light, right_light, left_light

# --- Scene setup ---
ddg.blender.scene.clear(deep=True)
setup_area_lights(shadow=True)
cam = ddg.blender.camera.camera(location=(0, 8, 0))
ddg.blender.camera.look_at_point(cam, (0, 0, 0))
bpy.context.scene.camera = cam
# - collections -
col = ddg.blender.collection.collection(name="darboux collection")
point_col = ddg.blender.collection.collection(name = "point collection")
frames_col = ddg.blender.collection.collection(name = "frames collection")

# --- Domain parameters setup ---
b = 10
eps = 1e-3
# --- Bounded domains for [-imf, inf]x[0, 2*pi] and [0, inf]x[0, 2*pi] ---
domains = [
    [[-b, b, False], [0, 2 * np.pi, True]],
    [[0, b, False], [0, 2 * np.pi, True]],
]

def random_unit_vectors(seed = None):
    """
    generate four random unit length 3 dim vectors 
    return them as a 4 x 3 matrix 
    """
    if seed == None:
        rng  = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    
    v = rng.standard_normal((4,3))
    v /= np.linalg.norm(v, axis = 1, keepdims=True)
    return v

def generate_section(seed = 100, seed_radii = None, theta = np.pi / 2):
    """
    construct 4 motions = rot + tr that move base point p to z_i
    z_i are of radius 0.5 to 3 chosen randomly
    first by rotation about the axis y at angle theta
    then translate the rotated point to match z_i
    generate 3plane of motions by these transformations
    return the section of S with this plane 
    """
    y = random_unit_vectors(seed)
    z_dirs = random_unit_vectors(seed + 1)
    rng_z = np.random.default_rng(seed+3)
    radii = rng_z.uniform(0.5, 3.0, size=4)       # lengths in [0.5, 3.0]
    z = radii[:, None] * z_dirs                 # scale each direction

    q0s = []
    q1s = []
    H = []
    for _, (yi, zi) in enumerate(zip(y, z)):
        r, q = find_se3(np.array([1, 0, 0]), yi, zi, theta)
        q0, q1 = se3_to_study(r, q)
        q0s.append(q0)
        q1s.append(q1)
        H.append(np.concatenate((q0, q1)))  # 8-vector Study coord
    
    A_matrix = np.vstack([np.concatenate((q0,q1)) for q0, q1 in zip(q0s, q1s)])
    A = ddg.geometry.subspace_from_rows(A_matrix)    
    I = ddg.geometry.intersect(S, A)
    N = ddg.to_smooth_net(I)  # I is a quadric surface in a 3-space inside RP^7
    return z, N, H

def _line_from_to(p, q):
    """
    creates a line segment from point p to q
    """
    f = lambda s: (1.0 - s) * np.asarray(p) + s * np.asarray(q)
    sn = ddg.nets.SmoothNet(f, [[0.0, 1.0, False]])
    return ddg.nets.sample_smooth_net(sn, sampling=[[2, 't']])

def draw_frame_from_study(h, p0, L=1.0, name="F", col=None):
    """
    convert frames to blender
    """
    r, t = se3_from_study(h)
    if r is None: return
    p = apply_se3(r, t, p0)                         # new origin
    axes = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
    mats = [yellow, purple, black]
    for a, mat, tag in zip(axes, mats, "xyz"):
        D = _line_from_to(p, p + L * rotation(a, r)[1:])  # rotate axis by r
        ddg.blender.convert(D, f"{name}_{tag}", mat, col)

def normalize(y, atol = 1e-12):
    """
    for vectors with llength < atol raise error
    otherwie return normalized vector
    """
    y = np.asarray(y, dtype = float)
    n = np.linalg.norm(y)
    if n < atol: 
        raise ValueError("zero length axis")
    return y / n

def find_se3(x, y, z, theta = 2 * np.pi / 3):
    """
    given points x,y,z of R3. we want to find a quaternionic rotation and translation, such that
    x is rotated about the axis y for \theta radians and then the result is translated to match z
    ---------------
    input: x,y,z 3D (numpy) arrays. y \neq 0. 
    output: r,q as 4D (numpy) arrays 
    """
    sin_t = np.sin(theta / 2)
    cos_t = np.cos(theta / 2)
    y = normalize(y)
    r = np.array([cos_t, sin_t * y[0], sin_t * y[1], sin_t * y[2]])
    q = np.array([0.0, *z]) - rotation(x, r)
    return r, q

def se3_to_study(r, q):
    """
    given a pair of quaternioins
    return the corresponding element of the study quadric
    """
    q1 = 0.5 * q_mul(q, r)
    return r, q1

def rotation(x, r):
    """
    given x and a rotational quaternion r. return the image of x after perfoming the rotation
    as 4d vectors
    """
    x = np.array([0.0, *x])
    return q_mul(r, q_mul(x, q_conj(r)))

def split_smooth_net(N):
    """
    takes a smooth net or a net collection as a parameter
    returns a list discrete net or a collection of discrete nets with infinite domains now bounded
    """
    if isinstance(N, ddg.nets.SmoothNet):
        N1 = ddg.nets.SmoothNet(N.fct, domains[0])
        D1 = ddg.nets.sample_smooth_net(N1, sampling=[[300, 't'], [300, 't']])
        return [D1]
    if isinstance(N, ddg.nets.NetCollection):
        smooth_nets = [n for n in N if isinstance(n, ddg.nets.SmoothNet)]
        discrete_nets = []
        for net in smooth_nets:
            N1 = ddg.nets.SmoothNet(net.fct, domains[1])
            D1 = ddg.nets.sample_smooth_net(N1, sampling=[[200, 't'], [200, 't']])
            discrete_nets.append(D1)
        return discrete_nets
    
def callback(a: int, mode_switch = True, theta = np.pi / 2):
    """
    mode_switch = True
    callback function to create a random darboux cyclide with integer index.
    creates a new one on each call and converts to blender object.
    
    mode_switch = Fals
    generate a subspace via predetermined motion generators
    convert the orbit to blender
    also convert the starting point, 4 image points of generator orbits
    and standard frame based on the starting point transformed by 4 generators of 3 section
    """
    ddg.blender.collection.clear([col, point_col, frames_col], deep = True)
    if a < 0 or a > 100:
        return
    if not mode_switch:
        N  = get_set(a+533)
        for i, d in enumerate(split_smooth_net(N)):
            name = f"motion surface index {a,i}"
            Ssurf = ddg.nets.compose(study_act_on_p(np.array([1, 0, 0])), d)
            ddg.blender.convert(Ssurf, name, red, col)
        return
    else:
        z, N, H = generate_section(seed = a, theta = theta)
        for i, d in enumerate(split_smooth_net(N)):
            name = f"motion surface index {a,i}"
            Ssurf = ddg.nets.compose(study_act_on_p(np.array([1, 0, 0])), d)
            ddg.blender.convert(Ssurf, name, red, col)
        for i, zi in enumerate(z):
            name = f"point {i}"
            point = ddg.geometry.subspace_from_affine_points([zi])
            ddg.blender.convert(point, name, blue, point_col, point_radius = 0.08)
        point = ddg.geometry.subspace_from_affine_points([np.array([1, 0, 0])])
        ddg.blender.convert(point, "starting point", green, point_col, point_radius = 0.08)

        for i, h in enumerate(H):
            draw_frame_from_study(h, p0, L=0.35, name=f"F{i}", col=frames_col)

ddg.blender.props.add_props_with_callback(
    callback,
    ("a", "mode_switch", "theta"),
    0,
    True,
    np.pi / 2,
)

# render setup
ddg.blender.render.setup_eevee_renderer()
# ddg.blender.render.set_world_background(color = (0, 0, 0, 0))
ddg.blender.render.set_film_transparency()
bpy.context.scene.render.resolution_x = 1500
bpy.context.scene.render.resolution_y = 1500

ddg.blender.render.set_render_output_images("/home/sasha_rk/darboux/pics", time=True)
# ddg.blender.render.render_animation(start = 0, end = frames - 1, camera=cam)   