import numpy as np
import matplotlib.pyplot as plt
import time

class PathRecorder:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Click and drag to draw a trajectory")
        self.path = []
        self.t = []
        self.line, = self.ax.plot([], [], marker='o')
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.t0 = time.time()

    def on_press(self, event):
        if event.button == 1:  # Left mouse button
            self.t0 = time.time()
            self.path = [(event.xdata, event.ydata)]
            self.t = [(0)]
            self.line.set_data(*zip(*self.path))
            self.fig.canvas.draw()

    def on_release(self, event):
        if event.button == 1:  # Left mouse button
            t = time.time() - self.t0
            self.path.append((event.xdata, event.ydata))
            self.t.append(t)
            self.line.set_data(*zip(*self.path))
            self.fig.canvas.draw()
        plt.close(self.fig)

    def on_motion(self, event):
        if event.button == 1:  # Left mouse button
            if event.inaxes == self.ax:
                t = time.time() - self.t0
                # self.path.append((t,event.xdata, event.ydata))
                self.path.append((event.xdata, event.ydata))
                self.t.append(t)
                self.line.set_data(*zip(*self.path))
                self.fig.canvas.draw()

    def get_path(self):
        return np.array(self.t), np.array(self.path)

if __name__ == "__main__":
    path_recorder = PathRecorder()
    plt.show()

    t, path = path_recorder.get_path()
    print("Recorded path:", path)
    print("Time", t)
    print("Data point num:", path.shape)
    print("Time point num:", t.shape)