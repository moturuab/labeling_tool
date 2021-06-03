# Inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/event_handling/image_slices_viewer.py

# Usage - if numpy file volume_i.npy exists:
# python scroller.py i
from __future__ import print_function

import skimage
import skimage.io as io
import os



import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import sys
import glob
from matplotlib.widgets import TextBox, Button, RadioButtons
from matplotlib.patches import Circle
import time

def read_png_volume(dir, transform=None):

    vol = []
    for i in range(len(os.listdir(dir))):
        a = io.imread(os.path.join(dir, "{}.png".format(i)), as_gray=True)[np.newaxis, ...]

        # a = a[:a.shape[1]]
        # if transform:
        #   a = transform(a)
        vol.append(a)

    return np.concatenate(vol, 0)


start = time.time()

# current volume number
volume_number = sys.argv[1]

UNDO = False
DONE = False
FINISH = False

fig = plt.figure(figsize=(11, 7))
ax = plt.subplot2grid((1,4), (0, 1),)
ay = plt.subplot2grid((1,4), (0, 2),)

plt.tight_layout()
# print(ax)

coords = plt.axes([0.09, 0.25, 0.15, 0.65])
coords_box = Button(coords, 'Points selected:\n', color='white', hovercolor='white')

undo = plt.axes([0.09, 0.09, 0.15, 0.1])
undo_but = Button(undo, 'UNDO', color='white', hovercolor='red')

case = plt.axes([0.72, 0.44, 0.2, 0.2])
case_but = RadioButtons(case, ('NO FOLLOW-UP', 'FOLLOW-UP', 'BIOPSY'), (False,))

done = plt.axes([0.72, 0.25, 0.2, 0.1])
done_but = Button(done, 'DONE', color='white', hovercolor='green')
done.set_visible(False)

q1 = plt.axes([0.15, 0.95, 0.7, 0.03])
q1_but = Button(q1, 'Q1. How mentally demanding was the task?', color='white', hovercolor='white')
q1.set_visible(False)

q1a = plt.axes([0.15, 0.8, 0.7, 0.15])
q1a_but = RadioButtons(q1a, ['1', '2', '3', '4', '5'], (False,))
q1a.set_visible(False)

q2 = plt.axes([0.15, 0.77, 0.7, 0.03])
q2_but = Button(q2, 'Q2. How hurried or rushed was the pace of the task?', color='white', hovercolor='white')
q2.set_visible(False)

q2a = plt.axes([0.15, 0.62, 0.7, 0.15])
q2a_but = RadioButtons(q2a, ('1', '2', '3', '4', '5'), (False,))
q2a.set_visible(False)

q3 = plt.axes([0.15, 0.59, 0.7, 0.03])
q3_but = Button(q3, 'Q3. How successful were you in accomplishing what you were asked to do?', color='white', hovercolor='white')
q3.set_visible(False)

q3a = plt.axes([0.15, 0.44, 0.7, 0.15])
q3a_but = RadioButtons(q3a, ('1', '2', '3', '4', '5'), (False,))
q3a.set_visible(False)

q4 = plt.axes([0.15, 0.41, 0.7, 0.03])
q4_but = Button(q4, 'Q4. How hard did you have to work to accomplish your level of performance?', color='white', hovercolor='white')
q4.set_visible(False)

q4a = plt.axes([0.15, 0.26, 0.7, 0.15])
q4a_but = RadioButtons(q4a, ('1', '2', '3', '4', '5'), (False,))
q4a.set_visible(False)

q5 = plt.axes([0.15, 0.23, 0.7, 0.03])
q5_but = Button(q5, 'Q5. How insecure, discouraged, irritated, stressed, and annoyed were you?', color='white', hovercolor='white')
q5.set_visible(False)

q5a = plt.axes([0.15, 0.08, 0.7, 0.15])
q5a_but = RadioButtons(q5a, ('1', '2', '3', '4', '5'), (False,))
q5a.set_visible(False)

finish = plt.axes([0.3, 0.02, 0.4, 0.05])
finish_but = Button(finish, 'FINISH', color='white', hovercolor='green')
finish.set_visible(False)

class Labels():
    def __init__(self, volume_n):
        self.volume_n = volume_n

    def case(self, label):
        done.set_visible(True)
        fig.canvas.draw_idle()

    def done(self, label):
        coords.set_visible(False)
        undo.set_visible(False)
        case.set_visible(False)
        done.set_visible(False)
        ax.set_visible(False)
        q1.set_visible(True)
        q1a.set_visible(True)
        q2.set_visible(True)
        q2a.set_visible(True)
        q3.set_visible(True)
        q3a.set_visible(True)
        q4.set_visible(True)
        q4a.set_visible(True)
        q5.set_visible(True)
        q5a.set_visible(True)
        finish.set_visible(True)
        global DONE
        DONE = True
        fig.canvas.draw_idle()

    def finish(self, label):
        global FINISH
        FINISH = True
        fig.canvas.draw_idle()

    def undo(self, label):
        global UNDO
        UNDO = True
        fig.canvas.draw_idle()


# handle scrolling through volume
class IndexTracker(object):
    def __init__(self, ax, ay, X, Y, n):
        self.ax = ax
        ax.set_title('scrolling through VOLUME {}\n'.format(n))

        self.ay = ay

        self.X = X
        self.Y = Y
        rows, cols, self.slices = X.shape
        self.ind = 0
        self.points = []
        self.circles = []
        self.press = False
        self.move = False

        self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=0, vmax=1)
        self.mask = ay.imshow(self.Y[:, :, self.ind], cmap='gray', vmin=0, vmax=1)
        
        self.update()

    def onscroll(self, event):
        if event.button == 'up' and self.ind < self.slices - 1:
            self.ind = self.ind + 1
        elif event.button == 'down' and self.ind > 0:
            self.ind = self.ind - 1
        self.update()

    def update(self):
        if not DONE:
            self.im.set_data(self.X[:, :, self.ind])
            self.mask.set_data(self.Y[:, :, self.ind])

            for (point, circ) in zip(self.points, self.circles):
                if self.ind != point[2]:
                    circ.set_visible(False)
                else:
                    circ.set_visible(True)
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()

    def onclick(self, click):
        global UNDO
        if UNDO:
            self.circles[-1].set_visible(False)
            del self.points[-1]
            del self.circles[-1]
            UNDO = False
            self.im.axes.figure.canvas.draw()

        global DONE
        global FINISH
        if DONE and FINISH and q1a_but.value_selected is not None and q2a_but.value_selected is not None and q3a_but.value_selected is not None and q4a_but.value_selected is not None and q5a_but.value_selected is not None:
            f = open('labels_' + str(volume_number) + '.txt', 'w')

            f.write('VOLUME ' + str(volume_number))
            f.write('\n')
            global start
            end = time.time()
            f.write('Total time:\n')
            f.write(str(end - start))
            f.write(' seconds\n')
            s = 'Points selected:\n'
            for x in self.points:
                s += '[' + str(int(x[0])) + ',' + str(int(x[1])) + ',' + str(int(x[2])) + ']\n'
            f.write(s)
            f.write('Exam feedback:\n')
            f.write(case_but.value_selected)
            s = '\nQuestionnaire answers:\n'
            s += q1a_but.value_selected + ',' + q2a_but.value_selected + ',' + q3a_but.value_selected + ',' + q4a_but.value_selected + ',' + q5a_but.value_selected
            f.write(s)
            f.close()
            sys.exit()

        if self.press and not self.move:
            self.point = (click.xdata, click.ydata)
            if self.point != (None, None) and int(self.point[0]) > 1 and int(self.point[1]) > 1:
                self.points.append([self.point[0], self.point[1], self.ind])
                circ = Circle((int(self.point[0]), int(self.point[1]), self.ind), 20, fill=False, edgecolor='red', lw=2)
                self.circles.append(circ)
                self.ax.add_patch(circ)
                s = 'Points selected:\n'
                for x in self.points:
                    s += '[' + str(int(x[0])) + ',' + str(int(x[1])) + ',' + str(int(x[2])) + ']\n'
                coords_box.label.set_text(s)
                self.update()
            elif self.point != (None, None):
                s = 'Points selected:\n'
                for x in self.points:
                    s += '[' + str(int(x[0])) + ',' + str(int(x[1])) + ',' + str(int(x[2])) + ']\n'
                coords_box.label.set_text(s)
                self.update()
            return self.point

    # source: https://stackoverflow.com/questions/48446351/distinguish-button-press-event-from-drag-and-zoom-clicks-in-matplotlib
    def onpress(self, event):
        self.press = True

    def onmove(self, event):
        if self.press:
            self.move = True

    def onrelease(self, event):
        if self.press and not self.move:
            self.onclick(event)
        self.press=False; self.move=False


#X = np.load('/home/abhishekmoturu/Desktop/gan_cancer_detection/brain_mri_512/volume_{}.npy'.format(volume_number)).astype(np.float32)

X = read_png_volume("../wbmri/png/volume_{}".format(sys.argv[1])) / 255

X = np.moveaxis(X, 0, 2)

Y = np.random.randn(1024, 256, 64)
# Y = read_png_volume("../wbmri/masks/volume_{}".format(sys.argv[1])) / 255
Y = np.moveaxis(Y, 0, 2)

# X = np.random.randn(1024, 256, 64)
# X = np.random.randn(1024, 256, 64)

label = Labels(volume_number)
tracker = IndexTracker(ax, ay, X, Y, volume_number)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('button_press_event', tracker.onclick)
fig.canvas.mpl_connect('button_press_event', tracker.onpress)
fig.canvas.mpl_connect('button_release_event', tracker.onrelease)
fig.canvas.mpl_connect('motion_notify_event', tracker.onmove)

case_but.on_clicked(label.case)
done_but.on_clicked(label.done)
undo_but.on_clicked(label.undo)
finish_but.on_clicked(label.finish)

plt.show()
print("done")
