class AimLogger:
    def __init__(self, run):
        self.run = run

    def track(self, step, **metrics):
        for name, val in metrics.items():
            if val is not None:
                self.run.track(val, name=name, step=step)
