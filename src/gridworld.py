import numpy as np


class GridWorld:
    def __init__(self):
        self.grid_size = 5
        self.actions = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
        self.wind = 0.1
        self.gamma = 0.9

        self.num_states = self.grid_size ** 2
        self.num_actions = len(self.actions)

        # Calculate the transition probability matrix. (states, states, actions).
        self.P = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.num_actions)]
              for j in range(self.num_states)]
             for i in range(self.num_states)])

        self.initial_state = (0, 0)
        self.goal_state = (4, 4)
        self.barriers = ((2, 0), (2, 1), (2, 2), (1, 2), (0, 2))

    def step(self, s, a):
        """
        Given state and action, calculate the nest state.

        :param s: start state. int.
        :param a: action to take. int.
        :return: next state. int.
        """
        ns = np.random.choice(self.num_states, p=self.P[s, :, a])
        if ns == self.state_to_int(self.goal_state):
            done = 1
        elif (s != self.state_to_int(self.initial_state)) and (ns == self.state_to_int(self.initial_state)):
            done = 1
        else:
            done = 0
        return ns, done

    def int_to_state(self, i):
        """
        Convert a state int into the corresponding coordinate.

        i: State int.
        -> (x, y) int tuple.
        """

        return i // self.grid_size, i % self.grid_size

    def state_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.

        p: (x, y) tuple.
        -> State int.
        """

        return p[0] * self.grid_size + p[1]

    def _is_neighbour(self, s1, s2):
        """
        Judge if state_1 and state_2 are neighbours.

        :param s1: state_1, int.
        :param s2: state_2, int.
        :return: if state_1 and state_2 are neighbours. bool.
        """
        x1, y1 = self.int_to_state(s1)
        x2, y2 = self.int_to_state(s2)

        return np.abs(x1 - x2) + np.abs(y1 - y2) <= 1

    def _is_corner(self, s):
        """
        Judge if a state is in the corner.

        :param s: state. int.
        :return: if the state is in the corner. bool.
        """
        x, y = self.int_to_state(s)
        return (x, y) in ((0, 0), (0, self.grid_size-1),
                          (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1))

    def _is_edge(self, s):
        """
        Judge if a state in on the edge (INCLUDING CORNER!).

        :param s: state. int.
        :return: if the state is on the edge. bool.
        """
        x, y = self.int_to_state(s)

        return (x in (0, self.grid_size-1)) or (y in (0, self.grid_size-1))

    def _is_in_the_world(self, x, y):
        """
        Judge if a stage is inside the enviroment.

        :param x: x coordinate of state. int.
        :param y: y coordinate of state. int.
        :return: if the state is inside the enviroment. bool.
        """

        return (0 <= x < self.grid_size) and (0 <= y < self.grid_size)

    def _transition_probability(self, s, st, a):
        """
        Calculate the transition probability p(st | s, a)

        :param s: initial state. int.
        :param st: target state. int.
        :param a: action to take. int.
        :return: transition probability. float
        """

        if not self._is_neighbour(s, st):
            return 0.0

        x, y = self.int_to_state(s)
        xt, yt = self.int_to_state(st)
        dx, dy = self.actions[a]

        if not self._is_edge(s):
            if (x+dx, y+dy) == (xt, yt):
                return 1-self.wind + self.wind / self.num_actions
            else:
                return self.wind / self.num_actions

        else:
            if self._is_corner(s):
                if s == st:
                    if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                        return 1-self.wind + 3 * self.wind / self.num_actions
                    else:
                        return 3 * self.wind / self.num_actions
                else:
                    if (x+dx, y+dy) == (xt, yt):
                        return 1-self.wind + self.wind / self.num_actions
                    else:
                        return self.wind / self.num_actions
            else:
                if s == st:
                    if (dx, dy) == (0, 0) or not self._is_in_the_world(x+dx, y+dy):
                        return 1-self.wind + 2 * self.wind / self.num_actions
                    else:
                        return 2 * self.wind / self.num_actions
                else:
                    if (x + dx, y + dy) == (xt, yt):
                        return 1 - self.wind + self.wind / self.num_actions
                    else:
                        return self.wind / self.num_actions

