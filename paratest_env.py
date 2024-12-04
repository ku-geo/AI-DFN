from typing import Optional
import chex
import jax
from jax import random
from jax import numpy as jnp
from functools import partial
import matplotlib.pyplot as plt
from matplotlib import font_manager


@chex.dataclass
class ParaState:
    parameters_test: chex.Array
    target_length: chex.Array
    parameters_number: chex.Array
    resolution: chex.Array
    reward: chex.Array
    legal_action_mask: chex.Array
    change_range: chex.Array
    Upper: chex.Array
    lower: chex.Array
    observation_area: chex.Array
    key: chex.PRNGKey
    info: chex.PRNGKey
    terminated: chex.Array = jnp.bool_(False)
    truncated: chex.Array = jnp.bool_(False)

    def result_plt(self, save=None):

        n = 500 * self.parameters_test[0]
        batam = 90.0
        phim = 90.0
        k = 2.0
        sl = 0.5
        rm = 0.03 * self.parameters_test[1]
        testpoints = gendisc(self.key, n, batam, phim, k, sl, rm)
        p1, d1 = endpoints(testpoints, deep=0, area=self.observation_area)
        p2, d2 = endpoints(testpoints, deep=self.observation_area * 1, area=self.observation_area)
        p3, d3 = endpoints(testpoints, deep=self.observation_area * 0.5, area=self.observation_area)
        d = jnp.concatenate([d1, d2, d3], axis=0)
        klname = f"random_sa/kl_{self.key}.png"
        kl_plot(self.target_length, d, save, klname)
        for i, p in enumerate([p1, p2, p3]):
            p21name = f"random_sa/p21_{i}_{self.key}.png"
            fracture_map(p, self.observation_area, save, p21name)


class Paratest:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
            self,
            *,
            render_mode: Optional[str] = None,
            parameters_number: int = jnp.int32(2),
            target: chex.Array = None,
            observation_area: chex.Array = None,
    ):
        self.render_mode = render_mode
        self.parameters_number = parameters_number
        self.target = target
        self.observation_area = observation_area
        self.upper = jnp.array([1, 1])
        self.lower = jnp.array([0.1, 0.1])

    def reset(self, rng_key: chex.PRNGKey, test_parameters: chex.Array) -> ParaState:
        """
        :type rng_key: rngkey in jax
        """
        state = partial(_init, parameters_number=self.parameters_number,
                        target=self.target, test_parameters=test_parameters,
                        observation_area=self.observation_area)(key=rng_key)
        state = partial(_update)(state)
        return state

    def step(self, state: ParaState, action: chex.Array, key: Optional[chex.Array] = None) -> ParaState:
        del key
        assert isinstance(state, ParaState)
        state = partial(_step)(state, action)
        state = partial(_update)(state)
        return state


def _init(key: chex.PRNGKey, target: chex.Array, parameters_number: int,
          test_parameters: chex.Array, observation_area: chex.Array) -> ParaState:
    return ParaState(
        parameters_number=parameters_number,
        target_length=target,
        key=key,
        parameters_test=test_parameters,
        change_range=jnp.ones(parameters_number)*0.25,
        observation_area=observation_area,
        reward=jnp.zeros(1),
        resolution=jnp.array(0.01),
        legal_action_mask=jnp.ones(parameters_number * 3, dtype=jnp.bool_),
        Upper=jnp.array([4, 4]),
        lower=jnp.array([0.1, 0.1]),
        info=jnp.array([0.1, 0.1]),
    )


def _legal_action_mask(test_parameters, change_range, Upper, Lower) -> chex.Array:
    mask_lower = jnp.where(test_parameters-change_range < Lower, jnp.bool(False), jnp.bool(True))
    mask_upper = jnp.where(test_parameters + change_range > Upper, jnp.bool(False), jnp.bool(True))
    mask_mid = jnp.ones_like(mask_lower)
    legal_mask = jnp.ravel(jnp.vstack((mask_lower, mask_mid, mask_upper)).T)
    return legal_mask



def _update(state: ParaState) -> ParaState:
    state = state.replace(legal_action_mask=state.legal_action_mask*_legal_action_mask(
        test_parameters=state.parameters_test,
        change_range=state.change_range,
        Upper=state.Upper,
        Lower=state.lower,
    ))
    kl_value, p21_value, rewardfuction = reward(parameters_test=state.parameters_test,
                                         targets_endpoints_length=state.target_length,
                                         observation_area=state.observation_area,rngkey=state.key)
    state = state.replace(reward=rewardfuction)
    state = state.replace(info=jnp.array([kl_value, p21_value]))
    state = state.replace(terminated=jnp.where(state.reward > jnp.float32(80), jnp.bool_(True), jnp.bool_(False)))
    return state


def _step(state: ParaState, action: jnp.integer) -> ParaState:
    assert isinstance(state, ParaState)
    _para = action // 3
    _change = action % 3 - 1
    state = state.replace(parameters_test=state.parameters_test.at[_para].set(
        jnp.where(_change, state.parameters_test[_para] + _change * state.change_range[_para],
                  state.parameters_test[_para])))
    state = state.replace(change_range=state.change_range.at[_para].set(
        jnp.where(_change, state.change_range[_para], state.change_range[_para] / 2)
    ))
    state = state.replace(legal_action_mask=state.legal_action_mask.at[action].set(
        jnp.where(((_change == 0) & (state.change_range[_para] < state.resolution)),
                  ~state.legal_action_mask[action], state.legal_action_mask[action])
    ))
    state = state.replace(legal_action_mask=state.legal_action_mask.at[action - 2 * _change].set(
        jnp.where(_change == 0, state.legal_action_mask[action - 2 * _change],
                  jnp.bool_(False))
    ))
    state = state.replace(legal_action_mask=state.legal_action_mask.at[_para * 3].set(
        jnp.where(_change == 0, jnp.bool_(True),
                  state.legal_action_mask[_para * 3])
    ))
    state = state.replace(legal_action_mask=state.legal_action_mask.at[_para * 3 + 2].set(
        jnp.where(_change == 0, jnp.bool_(True),
                  state.legal_action_mask[_para * 3 + 2])
    ))
    return state


def env_test(a, env, action) -> ParaState:
    a = jax.vmap(env.step)(a, action.astype(int))
    print(a.change_range[0,])
    print(a.legal_action_mask[0,])
    return a


def kl_divergence(targetdata: jnp.array, testdata: jnp.array) -> jnp.array:
    bin_edges1 = jnp.histogram_bin_edges(targetdata, bins=9)
    bin_edges1 = jnp.append(bin_edges1, jnp.array([10000]))
    bin_edges1 = bin_edges1.at[0].set(jnp.zeros_like(bin_edges1[0])+0.01)
    hist1, _ = jnp.histogram(targetdata, bins=bin_edges1, density=True)
    hist2, _ = jnp.histogram(testdata, bins=bin_edges1, density=True)
    hist1_normalized = hist1 / jnp.sum(hist1)
    hist2_normalized = hist2 / jnp.sum(hist2)
    kl_div = jnp.sum(hist1_normalized * (jnp.log(hist1_normalized + 1e-10) - jnp.log(hist2_normalized + 1e-10)))
    return kl_div


def kl_plot(targetdata: jnp.array, testdata: jnp.array, save=None, kl_savename=None) -> jnp.array:
    # file_path = 'data_1.pkl'
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_prop = font_manager.FontProperties(fname=font_path)
    bin_edges1 = jnp.histogram_bin_edges(targetdata, bins=9)
    bin_edges1 = jnp.append(bin_edges1, jnp.array([10000]))
    bin_edges1 = bin_edges1.at[0].set(jnp.zeros_like(bin_edges1[0])+0.01)
    hist1, _ = jnp.histogram(targetdata, bins=bin_edges1, density=True)
    hist2, _ = jnp.histogram(testdata, bins=bin_edges1, density=True)
    hist1_normalized = hist1 / jnp.sum(hist1)
    hist2_normalized = hist2 / jnp.sum(hist2)
    # kl_div = jnp.sum(hist1_normalized * (jnp.log(hist1_normalized + 1e-10) - jnp.log(hist2_normalized + 1e-10)))
    fig1=plt.figure(figsize=(8, 5))
    num_bins = len(hist1_normalized)
    # bar_width = 1.0 / num_bins
    positions = jnp.arange(num_bins)
    plt.bar(positions, hist1_normalized, color="#FA7F6F", align='edge', alpha=0.7, width=1.0, label='Observation of the surface', edgecolor='r')
    plt.bar(positions, hist2_normalized, color="#BEB8DC", align='edge',alpha=0.5, width=1.0, label="Slice of simulated rock", edgecolor="#BEB8DC")
    plt.xlabel('Length (m)', fontproperties=font_prop)
    plt.xlim([0,10])
    bin_labels = [f"{edge:.2f}" for edge in bin_edges1]
    bin_labels[0]="0"
    plt.xticks(positions, bin_labels[:-1])
    plt.ylabel('Frequency',fontproperties=font_prop)
    plt.legend(prop=font_prop)
    # plt.title('Histogram Comparison')
    if save is not None:
        fig1.savefig(kl_savename, dpi=300)
    else:
        plt.show()


def fracture_map(points, area, save, p21_savename):
    fig2 = plt.figure()
    for line in points:
        x_values = [line[0], line[2]]
        y_values = [line[1], line[3]]
        plt.plot(x_values, y_values, color="black")
        plt.xlim([0,0.5])
        plt.ylim([0, 0.5])
    if save is not None:
        fig2.savefig(p21_savename, dpi=300)
    else:
        plt.show()
    # plt.show()


def p21_mae(P: jnp.array, Q: jnp.array, samlpe_num:int) -> jnp.array:
    return jnp.abs((jnp.sum(P) - jnp.sum(Q)/samlpe_num) / jnp.sum(P))


def gendisc(key, n, batam, phim, k, sl, rm):
    """
    Assume the discontinuity as Beacher disc,
    using rock statistical information (n, batam, phim, k).

    Parameters:
    key: PRNG key for JAX random functions
    n (int): Number of all discontinuities in simulated rock
    batam (float): Mean angle of all discontinuities (in degrees)
    phim (float): Mean direction of all discontinuities (in degrees)
    k (float): Index of fisher-distribution
    sl (float): Square length (size of the simulated rock)
    rm (float): Mean radius of all discontinuities

    Returns:
    jnp.ndarray: Array containing disc's spatial position, size and direction.
    """
    nmax = 2000
    n = jnp.floor(n).astype(int)
    batam = jnp.radians(batam)
    phim = jnp.radians(phim)

    A = jnp.array([
        [jnp.cos(batam) * jnp.cos(phim), -jnp.sin(phim), jnp.sin(batam) * jnp.cos(phim)],
        [jnp.cos(batam) * jnp.sin(phim), jnp.cos(phim), jnp.sin(batam) * jnp.sin(phim)],
        [-jnp.sin(batam), 0, jnp.cos(batam)]
    ])

    key1, key2, key3, key4, key5 = random.split(key, 5)

    x1 = random.uniform(key1, shape=(nmax,))
    bata = jnp.arccos(jnp.log(jnp.exp(k) - 2 * jnp.sinh(k) * x1) / k)
    x2 = random.uniform(key2, shape=(nmax,))
    phi = 2 * jnp.pi * x2

    def compute_B(bata, phi):
        B = jnp.dot(A, jnp.array([jnp.sin(bata) * jnp.cos(phi), jnp.sin(bata) * jnp.sin(phi), jnp.cos(bata)]))
        B = jnp.where(B[2] < 0, -B, B)
        bata = jnp.arccos(B[2])
        phi = jnp.where(bata == 0, 0, jnp.where(B[0] / jnp.sin(bata) >= 1, 0, jnp.arccos(B[0] / jnp.sin(bata))))
        phi = jnp.where(B[1] < 0, 2 * jnp.pi - phi, phi)
        return jnp.degrees(bata), jnp.degrees(phi)

    compute_B(bata, phi)
    D_bata_phi = jax.vmap(compute_B)(bata, phi)

    D = jnp.zeros((nmax, 6))
    D = D.at[:, 0].set(D_bata_phi[0])
    D = D.at[:, 1].set(D_bata_phi[1])
    # random_values = jax.random.uniform(key3, (nmax, 3), minval=0, maxval=1) * sl
    zero_mask = jax.random.bernoulli(key5, n/nmax, (nmax, 1))
    D = D.at[:, 2:5].set(random.uniform(key3, shape=(nmax, 3)) * sl)
    # D = D.at[:, 5].set(random.exponential(key4, shape=(nmax,)) * rm)
    # power rate distribution alpha=2.5
    alpha = 2.5
    xmin = rm /((alpha -1 )/(alpha - 2))
    u = random.uniform(key4, shape=(nmax,))

    lognormal_mean = rm
    lognormal_var = 0.1

    mu = jnp.log(lognormal_mean ** 2 / jnp.sqrt(lognormal_var + lognormal_mean ** 2))
    sigma = jnp.sqrt(jnp.log(1 + lognormal_var / lognormal_mean ** 2))
    normal_samples = jax.random.normal(key4, shape=(nmax,))
    lognormal_samples = jnp.exp(mu + sigma * normal_samples)
    D = D.at[:, 5].set(lognormal_samples)

    # D = D.at[:, 5].set(xmin * (1 - u) ** (1 /(1 - alpha)))
    return D*zero_mask


def endpoints(Fractures: jnp.ndarray, deep, area):
    def compute_endpoints(fracture: jnp.ndarray):
        bata = fracture[0] / 180 * jnp.pi
        phi = fracture[1] / 180 * jnp.pi
        x0 = fracture[2]
        y0 = fracture[3] - deep
        z0 = fracture[4]
        r = fracture[5]
        r_corrected = jnp.sqrt(jnp.maximum(0, r ** 2 - y0 ** 2))
        a = jnp.cos(phi) * jnp.sin(bata)
        b = jnp.sin(phi) * jnp.sin(bata)
        c = jnp.cos(bata)
        sigma = 4 * b ** 2 * c ** 2 * y0 ** 2 - 4 * (a ** 2 + c ** 2) * (b ** 2 * y0 ** 2 - a ** 2 * r_corrected ** 2)
        z1 = jnp.where(
            sigma >= 0,
            (2 * b * c * y0 + jnp.sqrt(sigma)) / (2 * (a ** 2 + c ** 2)),
            0.
        )
        z2 = jnp.where(
            sigma >= 0,
            (2 * b * c * y0 - jnp.sqrt(sigma)) / (2 * (a ** 2 + c ** 2)),
            0.
        )
        x1 = jnp.where(
            jnp.abs(a) > 1e-5,
            b / a * y0 - c / a * z1,
            jnp.where(c != 0, jnp.sqrt(jnp.maximum(0, r_corrected ** 2 - (b / c * y0) ** 2)), 0.)
        )
        x2 = jnp.where(
            jnp.abs(a) > 1e-5,
            b / a * y0 - c / a * z2,
            jnp.where(c != 0, -jnp.sqrt(jnp.maximum(0, r_corrected ** 2 - (b / c * y0) ** 2)), 0.)
        )
        z1 = jnp.where(jnp.abs(a) <= 1e-5, b / c * y0, z1)
        z2 = jnp.where(jnp.abs(a) <= 1e-5, b / c * y0, z2)
        x1 = jnp.where(r ** 2 - y0 ** 2 < 0, 0., x1 + x0)
        x2 = jnp.where(r ** 2 - y0 ** 2 < 0, 0., x2 + x0)
        z1 = jnp.where(r ** 2 - y0 ** 2 < 0, 0., z1 + z0)
        z2 = jnp.where(r ** 2 - y0 ** 2 < 0, 0., z2 + z0)

        def point_isnan(six_points):
            # six_points = six_points.flatten()
            min_four_indices = jnp.argsort(six_points[:, 0])[:2]
            return six_points[min_four_indices, :].flatten()


        dx = x2 - x1
        dz = z2 - z1
        m = jnp.where(dx != 0, dz / dx, jnp.inf)
        c = z1 - m * x1

        def is_between(val1, val2, val):
            return jnp.logical_and(jnp.minimum(val1, val2) <= val, val <= jnp.maximum(val1, val2))

        nan_point = jnp.array([jnp.inf, jnp.inf])

        def intersect_line(x, z):
            return jnp.where(is_between(x1, x2, x) & is_between(z1, z2, z), jnp.array([x, z]), nan_point)

        def box_inside(arr):
            x = arr[0]
            z = arr[1]
            return jnp.where(is_between(0, area, x) & is_between(0, area, z), jnp.array([x, z]), nan_point)

        points = jnp.array([
            box_inside(intersect_line(0, c)),
            box_inside(intersect_line(area, m * area + c)),
            box_inside(intersect_line((area - c) / m, area)),
            box_inside(intersect_line(-c / m, 0)),
            box_inside(jnp.array([x1, z1])),
            box_inside(jnp.array([x2, z2]))
        ])
        result = point_isnan(points)

        point = jnp.where(result[0] == jnp.inf, jnp.array([0., 0, 0, 0]), result)
        return point

    _all_points = jax.vmap(compute_endpoints)(Fractures)
    point1 = _all_points[:, :2]
    point2 = _all_points[:, 2:]
    distances = jnp.linalg.norm(point1 - point2, axis=1)

    return _all_points, distances


def reward(parameters_test, targets_endpoints_length, observation_area, rngkey, orientation=None,):
    n = 500 * parameters_test[0]
    batam = 90.0
    phim = 0.0
    k = 2.0
    sl = 0.5
    rm = 0.03 * parameters_test[1]
    key2 = rngkey
    testpoints = gendisc(key2, n, batam, phim, k, sl, rm)
    p1, d1 = endpoints(testpoints, deep=0, area=observation_area)
    p2, d2 = endpoints(testpoints, deep=observation_area*1, area=observation_area)
    p3, d3 = endpoints(testpoints, deep=observation_area*0.5, area=observation_area)
    # p = jnp.concatenate([p1, p2, p3], axis=0)
    d = jnp.concatenate([d1, d2, d3], axis=0)
    kl_value = kl_divergence(targets_endpoints_length, d)
    p21_value = p21_mae(targets_endpoints_length, d, 3)
    red = kl_value + p21_value
    # red = p21_value
    rewardfuction = 100/jnp.cosh(10*red)
    return kl_value, p21_value, rewardfuction
