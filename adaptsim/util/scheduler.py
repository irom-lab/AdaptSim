class _scheduler(object):

    def __init__(self, last_epoch=-1, verbose=False):
        self.cnt = last_epoch
        self.verbose = verbose
        self.variable = None
        self.step()

    def step(self):
        self.cnt += 1
        value = self.get_value()
        self.variable = value

    def get_value(self):
        raise NotImplementedError

    def get_variable(self):
        return self.variable


class StepLRFixed(_scheduler):

    def __init__(
        self, init_value, period, end_value, step_size=0.1, last_epoch=-1,
        verbose=False
    ):
        self.init_value = init_value
        self.period = period
        self.step_size = step_size
        self.end_value = end_value
        super(StepLRFixed, self).__init__(last_epoch, verbose)

    def get_value(self):
        if self.cnt == 0:
            return self.init_value
        elif self.cnt > self.period:
            self.cnt = 0
            if self.step_size > 0:
                self.variable = min(
                    self.end_value, self.variable + self.step_size
                )
            else:
                self.variable = max(
                    self.end_value, self.variable + self.step_size
                )
        return self.variable