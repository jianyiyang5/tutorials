import time
import math
import matplotlib.pyplot as plt

# To keep track of how long training takes I am adding a
# ``timeSince(timestamp)`` function which returns a human readable string:
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def plot_losses(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()