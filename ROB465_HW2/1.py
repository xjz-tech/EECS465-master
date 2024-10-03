import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx
import os.path
import time
import sys

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


class StateIndices:
    X = 0
    Y = 1
    HEADING = 2


class ControlIndices:
    SPEED = 0
    TURN = 1


class CarEnvironment:
    """Simplified Dubin's car environment"""

    def __init__(self):
        # plotting
        self.fig = plt.gcf()
        self.ax = plt.gca()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal')
        self.car_artists = {}

        self.fig.suptitle('Press Space Bar to End Run', fontsize=14, fontweight='bold')

        # set up the space bar call back for ending early
        def onpress(event):
            global bKeepRunning

            if event.key == ' ':
                if bKeepRunning == False:
                    # plot is already stopped, exit
                    exit()
                else:
                    bKeepRunning = False
                    print('Ending the run')
                    # self.fig.suptitle('Press Space Bar to Close Program', fontsize=14, fontweight='bold', color='red')

        self.fig.canvas.mpl_connect('key_press_event', onpress)

        plt.ion()
        plt.show()
        self.state_history = []

    def visualize_state(self, state, name='current', color=(0, 0, 0), plot_trail=True):
        """Draw a state"""
        car_artists = self.car_artists.get(name, [])
        for artist in car_artists:
            artist.remove()
        car_artists = []

        if plot_trail and name == 'current':
            self.state_history.append(state)
            xs, ys, _ = zip(*self.state_history[-2:])
            self.ax.plot(xs, ys, c=(0, 0, 1), linestyle='dashed')

        # represent car with an arrow pointing along its heading
        arrow_length = 0.1
        c = np.cos(state[StateIndices.HEADING])
        s = np.sin(state[StateIndices.HEADING])
        dx = arrow_length * c
        dy = arrow_length * s
        car_artists.append(
            plt.arrow(state[StateIndices.X], state[StateIndices.Y], dx, dy, width=arrow_length / 10, ec=color, fc=color,
                      label=name))

        # plot rectangle instead of point to represent robot
        # car_artists.append(plt.scatter(state[StateIndices.X], state[StateIndices.Y], color=c, label=name))
        h = 0.1
        w = 0.15
        offset = np.array([-w / 2, -h / 2])
        rot = np.array([[c, -s],
                        [s, c]])
        offset = rot @ offset
        car_artists.append(plt.Rectangle((state[StateIndices.X] + offset[0],
                                          state[StateIndices.Y] + offset[1]), w, h,
                                         angle=state[StateIndices.HEADING] * 180 / np.pi,
                                         ec=color, fc=color, fill=False))
        self.ax.add_artist(car_artists[-1])

        self.car_artists[name] = car_artists
        if name == 'current':
            self.ax.legend()
        plt.pause(0.05)

    def save_plot(self, filename):
        plt.savefig(filename)

    def true_dynamics(self, state, control):
        """x_{t+1} = f(x_t, u_t) true dynamics function of the car"""
        """The true dynamics should be used as a 'black box' to simulate the car motion"""
        t = 0.1

        next_state = np.copy(state)
        next_state[StateIndices.X] += control[ControlIndices.SPEED] * t * np.cos(state[StateIndices.HEADING])
        next_state[StateIndices.Y] += control[ControlIndices.SPEED] * t * np.sin(state[StateIndices.HEADING])
        next_state[StateIndices.HEADING] += t * control[ControlIndices.TURN]
        return next_state


def linearize_dynamics_numerically(x_r, u_r, h, true_dynamics):
    """Numerically linearize car dynamics around reference state x_r and control u_r with parameter h
    Linear dynamics is of the form  x_{t+1} = Ax_t + Bu_t
    This function returns A, B

    In this problem we recommend using finite differences; this should be the last resort in practice since
    it is the least accurate, but the most robust since it can handle black box functions.
    You should use the true dynamics as a 'black box', i.e. your method should be able to handle any
    true_dynamics function given to it
    """

    # wrapper around function to have it take a single vector as input
    def f(combined_input):
        x = combined_input[:3]
        u = combined_input[3:]
        return true_dynamics(x, u)

    # combined reference point
    xu_r = np.r_[x_r, u_r]
    # function evaluated at reference point
    f_r = f(xu_r)

    m = f_r.shape[0]
    n = xu_r.shape[0]
    # Jacobian (m, n) where m is the output dimension, n is the input dimension
    Jacobian = np.zeros((m, n))

    # Implement Newton's difference quotient using h
    #### YOUR CODE HERE ####
    for i in range(m):
        for j in range(n):
            xu_r_new = xu_r.copy()
            xu_r_new[j] += h
            Jacobian[i, j] = (f(xu_r_new)[i] - f_r[i]) / h
    #### YOUR CODE HERE ####

    A = Jacobian[:, :3]  # the left half of the Jacobian is A
    B = Jacobian[:, 3:]  # the right half of the Jacobian is B
    return A, B


def linearize_dynamics(x_r, u_r, t):
    """Linearize car dynamics around reference state and control with sampling time t"""
    A = np.array([[1, 0, - u_r[ControlIndices.SPEED] * np.sin(x_r[StateIndices.HEADING]) * t],
                  [0, 1, u_r[ControlIndices.SPEED] * np.cos(x_r[StateIndices.HEADING]) * t],
                  [0, 0, 1]])
    B = np.array([[np.cos(x_r[StateIndices.HEADING]) * t, 0],
                  [np.sin(x_r[StateIndices.HEADING]) * t, 0],
                  [0, t]])
    return A, B


def optimize_single_action(goal_state, current_state, reference_control, A, B, speed_limit, turn_limit):
    """Optimize to get the best single step action under our linear dynamics

    :param A state dynamics matrix
    :param B control dynamics matrix
    :param speed_limit (min_speed, max_speed) for the control dimension; boundaries are allowed
    :param turn_limit (min_turn, max_turn) where the boundaries are allowed
    """

    # single step control
    # define a cvx.Variable for the control
    control = cvx.Variable(B.shape[1])

    # define the control constraints and the objective, then use cvxpy to solve the QP
    ### YOUR CODE HERE ###
    A_ = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    b = np.array([-speed_limit[0], speed_limit[1], -turn_limit[0], turn_limit[1]])
    P = B.T @ B
    q = goal_state - current_state + B @ reference_control

    prob = cvx.Problem(cvx.Minimize(cvx.quad_form(control, P) - 2 * q.T @ B @ control),
                       [A_ @ control <= b])

    # prob = cvx.Problem(cvx.Minimize(cvx.norm(goal_state - (current_state + A @ (current_state - current_state) + B @ (control - reference_control)))), [A_ @ control <= b])

    prob.solve()
    ### YOUR CODE HERE ###

    if control.value is not None:
        return control.value
    else:
        # control has not been computed
        return np.zeros(B.shape[1])


def run_problem(probname, start_state, goal_state,
                h_in=0.1, reference_control=np.array([0.5, 0.]), speed_limit=(0, 1), turn_limit=(-2, 2)):
    global bKeepRunning

    car = CarEnvironment()
    car.visualize_state(goal_state, name='goal', color=(0, 0.7, 0), plot_trail=False)

    prediction_error = []

    state = start_state.copy()
    car.visualize_state(state)
    tol = 1e-10
    bKeepRunning = True

    MaxSteps = 40

    for i in range(MaxSteps):
        if (not bKeepRunning):
            break
        # A, B = linearize_dynamics(state, reference_control, t=t)
        A, B = linearize_dynamics_numerically(state, reference_control, h=h_in, true_dynamics=car.true_dynamics)
        control = optimize_single_action(goal_state, state, reference_control, A, B, speed_limit, turn_limit)

        # use our linearized dynamics to predict next state under this control
        # predicted state should be state + (A @ state-state) + (B @ (control-reference_control)), because we linearize about the current state, which we assume is an equilibrium point and state - state = 0, we have:
        predicted_state = state + B @ (control - reference_control)

        # check that control is within bounds
        if not (speed_limit[0] - tol < control[ControlIndices.SPEED] < speed_limit[1] + tol
                and turn_limit[0] - tol < control[ControlIndices.TURN] < turn_limit[1] + tol):
            raise RuntimeError(
                "Control is out of bounds; optimization constraints may be incorrectly set"
                "\ncontrol: {}\nlimits: {} and {}".format(control, speed_limit, turn_limit))

        state = car.true_dynamics(state, control)
        prediction_error.append(state - predicted_state)
        # print(np.sum((goal_state - predicted_state)**2))
        car.visualize_state(predicted_state, name='predicted', color=(0.7, 0, 0), plot_trail=False)
        car.visualize_state(state)

        # if control is close to 0, end
        if (np.linalg.norm(control) < tol):
            break

    bKeepRunning = False
    car.fig.suptitle('Press Space Bar to Close Program', fontsize=14, fontweight='bold', color='red')
    # car.save_plot(os.path.join(SCRIPT_DIR, "{}.png".format(name)))

    print('Distance to goal:\n', np.linalg.norm((goal_state - state)))

    # prediction_error = np.stack(prediction_error)
    # print("max prediction error: {}".format(np.max(prediction_error)))


if __name__ == "__main__":

    args = sys.argv[1:]
    if (len(args) == 0):
        print("Specify what to run:")
        print("  'python3 carqp.py test_linearization' will test the numerical linearization method")
        print("  'python3 carqp.py run_test [test index]' will run the simulation for a specific test index (0-2)")
        exit()

    if args[0] == 'test_linearization':
        car = CarEnvironment()
        current_state = np.array([0.2, 0.1, 0.4])
        reference_control = np.array([0., 0.])
        test_control = np.array([0.1, 0.2])
        h = 0.01

        print('Testing linearization of dynamics for')
        print('  Current state:')
        print('   ', current_state)
        print('  Reference control:')
        print('   ', reference_control)
        print('  h:')
        print('   ', h)

        A, B = linearize_dynamics_numerically(current_state, reference_control, h=h, true_dynamics=car.true_dynamics)
        print('A:\n', A)
        print('B:\n', B)

        print('\nTest control:')
        print('   ', test_control)

        # get prediction
        predicted_nextx = current_state.T + A @ (current_state.T - current_state.T) + B @ (
                    test_control.T - reference_control.T)

        # get true next state
        true_nextx = car.true_dynamics(current_state, test_control)

        print('\nPredicted state using linearized dynamics Ax + Bu:')
        print('  ', predicted_nextx)
        print('True state (using true dynamics):')
        print('  ', true_nextx)
        print('Prediction error:')
        print('  ', np.linalg.norm(predicted_nextx - true_nextx))

        exit()

    elif args[0] == 'run_test':
        try:
            testind = int(args[1])
        except:
            print("ERROR: Test index has not been specified")
            exit()

        # state format is [x, y, theta]
        start_state = np.array([0., 0., 0.])
        goal_states = np.array([[0.6, 0.4, 1],  # test 0
                                [0.7, -0.6, -1],  # test 1
                                [0.8, -0.3, -1.5]])  # test 2

        run_problem(testind, start_state, goal_states[testind], h_in=0.01, reference_control=np.array([0.0, 0.0]))

        # wait until the plot closes
        plt.show(block=True)