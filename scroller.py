# Inspired by: https://github.com/matplotlib/matplotlib/blob/master/examples/event_handling/image_slices_viewer.py

# Usage - if numpy file volume_i.npy exists:
# python scroller.py i
from __future__ import print_function

import skimage
import skimage.io as io
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure
import sys
import skimage
import skimage.io as io
import os
import glob
from matplotlib.widgets import TextBox, Button, RadioButtons
from matplotlib.patches import Circle

import matplotlib.widgets as mpwidgets
import time
from datetime import datetime

def read_png_volume(dir, transform=None):
    vol = []
    for i in range(len(os.listdir(dir))):
        a = io.imread(os.path.join(dir, "{}.png".format(i)), as_gray=True)[np.newaxis, ...]
        vol.append(a)
    return np.concatenate(vol, 0)

def read_png_volume2(dir, transform=None):
    vol = []
    for i in range(len(os.listdir(dir))):
        a = io.imread(os.path.join(dir, "slice_{}.png".format(i)), as_gray=True)[np.newaxis, ...]
        # a = a[:a.shape[1]]
        # if transform:
        #   a = transform(a)
        vol.append(a)
    return np.concatenate(vol, 0)

global OPACITY
OPACITY = 0.5

def update(value):
    global OPACITY
    OPACITY = value
    tracker.im.set_alpha(value)
    fig.canvas.draw_idle()

start = time.time()
end = 0

# current volume number
volume_number = sys.argv[1]

UNDO = False
DONE = False
FINISH = False

fig = plt.figure(figsize=(9, 10))
ax = plt.subplot2grid((1,3), (0, 1),)

ay = plt.subplot2grid((14,11), (1, 9), colspan=2)

if int(volume_number) <= 25:
    slider0 = mpwidgets.Slider(ax=ay, label='opacity', valmin=0, valmax=1, valinit=OPACITY)
    slider0.on_changed(update)
else:
    ay.set_visible(False)

plt.subplots_adjust(top=0.9)
fig.tight_layout()

coords = plt.axes([0.034, 0.25, 0.15, 0.65])
coords_box = Button(coords, 'Points selected:\n', color='white', hovercolor='white')

undo = plt.axes([0.034, 0.09, 0.15, 0.1])
undo_but = Button(undo, 'UNDO', color='white', hovercolor='red')

case = plt.axes([0.772, 0.44, 0.2, 0.2])
case_but = RadioButtons(case, ('NO FOLLOW-UP', 'FOLLOW-UP', 'BIOPSY'), (False,))

done = plt.axes([0.772, 0.25, 0.2, 0.1])
done_but = Button(done, 'DONE', color='white', hovercolor='green')
done.set_visible(False)

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
        if int(volume_number) <= 25:
            slider0.set_active(False)
        ay.set_visible(False)

        finish.set_visible(True)

        self.q1 = plt.axes([0.15, 0.95, 0.7, 0.03])
        self.q1_but = Button(self.q1, 'Q1. How mentally demanding was the task? (1 - not at all, 5 - a great deal)', color='white', hovercolor='white')

        self.q1a = plt.axes([0.2, 0.8, 0.2, 0.15])
        self.q1a_but = RadioButtons(self.q1a, ['1', '2', '3', '4', '5'], (False,))
        self.q1a.axis('off')

        self.q2 = plt.axes([0.15, 0.77, 0.7, 0.03])
        self.q2_but = Button(self.q2, 'Q2. How hurried or rushed was the pace of the task?', color='white', hovercolor='white')

        self.q2a = plt.axes([0.2, 0.62, 0.2, 0.15])
        self.q2a_but = RadioButtons(self.q2a, ('1', '2', '3', '4', '5'), (False,))
        self.q2a.axis('off')

        self.q3 = plt.axes([0.15, 0.59, 0.7, 0.03])
        self.q3_but = Button(self.q3, 'Q3. How successful were you in accomplishing what you were asked to do?', color='white', hovercolor='white')

        self.q3a = plt.axes([0.2, 0.44, 0.2, 0.15])
        self.q3a_but = RadioButtons(self.q3a, ('1', '2', '3', '4', '5'), (False,))
        self.q3a.axis('off')

        self.q4 = plt.axes([0.15, 0.41, 0.7, 0.03])
        self.q4_but = Button(self.q4, 'Q4. How hard did you have to work to accomplish your level of performance?', color='white', hovercolor='white')

        self.q4a = plt.axes([0.2, 0.26, 0.2, 0.15])
        self.q4a_but = RadioButtons(self.q4a, ('1', '2', '3', '4', '5'), (False,))
        self.q4a.axis('off')

        self.q5 = plt.axes([0.15, 0.23, 0.7, 0.03])
        self.q5_but = Button(self.q5, 'Q5. How insecure, discouraged, irritated, stressed, and annoyed were you?', color='white', hovercolor='white')

        self.q5a = plt.axes([0.2, 0.08, 0.2, 0.15])
        self.q5a_but = RadioButtons(self.q5a, ('1', '2', '3', '4', '5'), (False,))
        self.q5a.axis('off')

        global DONE
        DONE = True
        fig.suptitle('')
        global end
        end = time.time()
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

    def __init__(self, label, ax, X, Y, n):
        self.ax = ax
        fig.suptitle('scrolling through VOLUME {}\n'.format(n))

        self.ay = ay
        self.label = label

        self.X = X
        self.Y = Y
        rows, cols, self.slices = X.shape
        self.ind = 0
        self.points = []
        self.circles = []
        self.press = False
        self.move = False
        self.ims = []
        self.masks = []
        for i in range(self.slices):
            self.ims.append(self.X[:, :, i])
            mask = self.Y[:, :, i]
            masked = np.ma.where(mask > 3 * np.mean(mask), 1, 0)
            masked = np.ma.masked_where(masked == 0, masked)
            self.masks.append(masked)
        ax.imshow(self.ims[self.ind], cmap='gray', vmin=0, vmax=1)
        #self.mask =
        #self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray', vmin=0, vmax=1)
        #self.mask = self.Y[:, :, self.ind]
        #masked = np.ma.where(self.mask > 3*np.mean(self.mask), 1, 0)
        #masked = np.ma.masked_where(masked == 0, masked)
        # https://matplotlib.org/stable/tutorials/colors/colormaps.html
        self.im = ax.imshow(self.masks[self.ind], cmap='bwr', interpolation='none', alpha=OPACITY, vmin=0, vmax=1)
        ax.autoscale(False)
        #self.mask = ay.imshow(self.Y[:, :, self.ind], cmap='gray', vmin=0, vmax=1)

        self.update()

    def onscroll(self, event):
        if event.button == 'up' and self.ind < self.slices - 1:
            self.ind = self.ind + 1
        elif event.button == 'down' and self.ind > 0:
            self.ind = self.ind - 1
        self.update()

    def onkeypress(self, event):
        sys.stdout.flush()
        if event.key == 'up' and self.ind < self.slices - 1:
            self.ind = self.ind + 1
        elif event.key == 'down' and self.ind > 0:
            self.ind = self.ind - 1
        elif event.key == 'left' or event.key == 'right':
            pass
        self.update()

    def update(self):
        if not DONE:
            #self.ax.cla()
            ax.imshow(self.ims[self.ind], cmap='gray', vmin=0, vmax=1)
            #self.im.set_data(self.X[:, :, self.ind])
            #self.im.set_cmap('gray')
            #self.im.set_clim(vmin=0)
            #self.im.set_clim(vmax=np.max(self.X[:,:,self.ind]))
            #self.mask = self.masks[self.ind] #self.Y[:, :, self.ind]
            #s = 1*(np.min(self.Y[:,:,self.ind]) + np.max(self.Y[:,:,self.ind]))/2
            #masked = np.ma.where(self.mask > 3*np.mean(self.mask), 1, 0)
            #masked = np.ma.masked_where(masked == 0, masked)
            #self.im.set_data(masked)
            #self.im = ax.imshow(self.masks[self.ind], cmap='bwr', alpha=OPACITY)
            #self.im.set_data(self.masks[self.ind])
            #self.im.set_cmap('bwr')
            #self.im.set_alpha(OPACITY)
            #self.im.set_data(self.masks[self.ind])
            #self.im.set_cmap('bwr')
            #self.mask.set_data(self.Y[:, :, self.ind])

            for (point, circ) in zip(self.points, self.circles):
                if self.ind != point[2]:
                    circ.set_visible(False)
                else:
                    self.ax.add_patch(circ)
                    circ.set_visible(True)
            ax.set_ylabel('slice %s' % self.ind)
            self.im.axes.figure.canvas.draw()
            #self.mask.axes.figure.canvas.draw()

    def onclick(self, click):
        global UNDO
        if UNDO and len(self.circles) > 0:
            self.circles[-1].set_visible(False)
            del self.points[-1]
            del self.circles[-1]
            UNDO = False
            self.im.axes.figure.canvas.draw()
            #self.mask.axes.figure.canvas.draw()

        global DONE
        global FINISH
        if DONE and FINISH and self.label.q1a_but.value_selected is not None and self.label.q2a_but.value_selected is not None and self.label.q3a_but.value_selected is not None and self.label.q4a_but.value_selected is not None and self.label.q5a_but.value_selected is not None:
            f = open('labels_' + str(volume_number) + '.txt', 'w')

            f.write('VOLUME ' + str(volume_number))
            f.write('\n')
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            f.write(dt_string)
            f.write('\n')
            global start
            global end
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
            s += self.label.q1a_but.value_selected + ',' + self.label.q2a_but.value_selected + ',' + self.label.q3a_but.value_selected + ',' + self.label.q4a_but.value_selected + ',' + self.label.q5a_but.value_selected
            f.write(s)
            f.close()
            sys.exit()

        if not DONE and self.press and not self.move:
            self.point = (click.xdata, click.ydata)
            if self.point != (None, None) and int(self.point[0]) > 1 and int(self.point[1]) > 1:
                self.points.append([self.point[0], self.point[1], self.ind])
                circ = Circle((int(self.point[0]), int(self.point[1]), self.ind), 25, fill=False, edgecolor='chartreuse', lw=2)
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

if len(sys.argv) < 3:
    X = read_png_volume("volumes/volume_{}".format(volume_number)) / 255.0
    X = np.moveaxis(X, 0, 2)
    if int(volume_number) <= 25:
        Y = read_png_volume("masks_final/volume_{}".format(volume_number)) / 50.0
        Y = np.moveaxis(Y, 0, 2)
    else:
        Y = np.zeros_like(X)
else:
    X = read_png_volume("nodule_im/volume_{}".format(volume_number)) / 255.0
    X = np.moveaxis(X, 0, 2)
    if int(volume_number) <= 25:
        Y = read_png_volume2("nodule_im_masks/volume_{}".format(volume_number)) / 50.0
        Y = np.moveaxis(Y, 0, 2)
    else:
        Y = np.zeros_like(X)

# print(Y.shape, X.shape, "------------")
# Y = np.repeat(X.copy()[:, :, :,np.newaxis], 3, axis=3)
# Y[:, :, :, :2] = 0

# mask = read_png_volume("../wbmri/png/volume_{}".format(sys.argv[1])) / 255


# X = np.random.randn(1024, 256, 64)
'''
X = np.random.rand(1024, 256, 64)
Y = np.zeros((1024, 256, 64))
print(X.min())
print(X.max())
Y[30:-30, 30:-30, :] = 0
Y[40:-40, 40:-40, :] = 0.25
Y[50:-50, 50:-50, :] = 0.5
Y[60:-60, 60:-60, :] = 0.75
Y[70:-70, 70:-70, :] = 1
print(Y.min())
print(Y.max())
'''

label = Labels(volume_number)
tracker = IndexTracker(label, ax, X, Y, volume_number)

fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
fig.canvas.mpl_connect('button_click_event', tracker.onclick)
fig.canvas.mpl_connect('button_press_event', tracker.onpress)
fig.canvas.mpl_connect('button_release_event', tracker.onrelease)
fig.canvas.mpl_connect('motion_notify_event', tracker.onmove)
fig.canvas.mpl_connect('key_press_event', tracker.onkeypress)

case_but.on_clicked(label.case)
done_but.on_clicked(label.done)
undo_but.on_clicked(label.undo)
finish_but.on_clicked(label.finish)

plt.show()
