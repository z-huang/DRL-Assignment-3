import matplotlib.pyplot as plt
import numpy as np


class Logger:
    def __init__(self, log_every=1, moving_avg_window=10):
        self.buffer = []
        self.values = []
        self.steps = []
        self.log_every = log_every
        self.moving_avg_window = moving_avg_window
        self._step_count = 0

    def step(self, value):
        self._step_count += 1
        self.buffer.append(value)

        if self._step_count % self.log_every == 0:
            avg = np.mean(self.buffer)
            self.values.append(avg)
            self.steps.append(self._step_count)
            self.buffer.clear()

    def plot(self, title="Logger", ylabel="Value", xlabel="Step", path=None):
        fig, ax = plt.subplots(figsize=(10, 5))

        values_np = np.array(self.values)
        ax.plot(self.steps, values_np, label='Value')

        if len(values_np) >= self.moving_avg_window:
            moving_avg = np.convolve(
                values_np,
                np.ones(self.moving_avg_window) / self.moving_avg_window,
                mode='valid'
            )
            steps_avg = self.steps[self.moving_avg_window - 1:]
            ax.plot(steps_avg, moving_avg,
                    label=f'Moving Avg ({self.moving_avg_window})')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

        if path is not None:
            fig.savefig(path)

        return fig
