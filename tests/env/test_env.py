from env import Environment
import numpy as np
from bike import Bike
from tqdm import tqdm


def test_env_init():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    assert test_env.g == -9.8
    assert test_env.elasticiity == 0
    assert test_env.x_max == 1000
    assert test_env.t_step == 0.01
    assert test_env.starting_positions[1] == 10
    assert test_env.starting_positions[0] == 0
    assert test_env.delta_x == 0.001
    np.testing.assert_array_equal(
        test_env.ground,
        np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    np.testing.assert_array_equal(
        test_env.ground_derivative,
        np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )


def test_set_bikes():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )

    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    assert test_env.n_bikes == 3
    assert test_env.bikes[0] == my_bike_1
    assert test_env.bikes[1] == my_bike_2
    assert test_env.bikes[2] == my_bike_3


def test_get_R():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    R = test_env.get_R()
    assert R is not None
    assert len(R) == 3


def test_get_K():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    K = test_env.get_K()
    assert K is not None
    assert len(K) == 3


def test_get_B():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    B = test_env.get_B()
    assert B is not None
    assert len(B) == 3


def test_get_Ms():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    Ms = test_env.get_Ms()
    assert Ms is not None
    assert len(Ms) == 3


def test_get_torks():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    torks = test_env.get_torks()
    assert torks is not None
    assert len(torks) == 3


def test_get_init_lenghs():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    init_lengths = test_env.get_init_lenghs()
    assert init_lengths is not None
    assert len(init_lengths) == 3


def test_get_Radius():
    test_env = Environment(
        g=-9.8,
        elasticiity=0,
        x_max=1000,
        t_step=0.01,
        starting_hight=10,
        delta_x=0.001,
        ground=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
        ground_derivative=np.array(
            [
                np.linspace(0, 1000, int(1000 / 0.001)),
                0 * np.linspace(0, 1000, int(1000 / 0.001)),
            ]
        ).T,
    )
    my_bike_1 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_2 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    my_bike_3 = Bike(
        1,
        1,
        2,
        3,
        1,
        4,
        1,
        2,
        3,
        0,
        1,
        3,
        2,
        4,
        3,
        2,
        3,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    test_env.set_bikes([my_bike_1, my_bike_2, my_bike_3])
    Radius = test_env.get_Radius()
    assert Radius is not None
    assert len(Radius) == 3


def test_get_connection_info():
    test_env = Environment()
    pos = np.array([[0, 0]])
    connection_info = test_env.get_connection_info(pos)
    assert connection_info is not None
