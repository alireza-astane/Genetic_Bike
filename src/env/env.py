# this is just an example of how we can write the abstract of the class and may change later !!!!!
class env:
    def __init__(self):
        # constructor
        self.trajectory = []
        self.n_bikes = 0

    def step(self):
        # step function
        pass

    def add_bikes(self, list_bikes):
        # add bikes
        pass

    def get_trajectory(self):
        # maybe someone wants to get the trajectory to visualize it
        return self.trajectory

    def evaluate(self):
        # evaluate the environment

        scores = np.zeros(self.n_bikes)
        return scores


# just wrtie the abstract of the class and fields and methods
# and then write the implementation afterward
# others will know which functions to call, what to input and what to expect from the class
# this is a good way to write code for collaboration
# I think we should write the abstracts frist, then we check with each other and then write the implementation and complete the empty functions
# this way we can divide the work and also know what to expect from each other
