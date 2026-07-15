import numpy as np
def LGP(img, r=1):
    padded = np.pad(img, (r, r), 'constant')
    a1 = padded[:-2*r, :-2*r]
    b1 = padded[:-2*r, r:-r]
    a2 = padded[:-2*r, 2*r:]
    b2 = padded[r:-r, 2*r:]
    a3 = padded[2*r:, 2*r:]
    b3 = padded[2*r:, r:-r]
    a4 = padded[2*r:, :-2*r]
    b4 = padded[r:-r, :-2*r]
    codes = (a1 >= a3) + 2*(a2 >= a4) + 4*(b1 >= b3) + 8*(b2 >= b4)
    return codes


