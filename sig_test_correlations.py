import numpy as np
def ind_btwn(r1, r2, n1, n2):
    L = r1 - r2 - 1.96 * np.sqrt((1-r1**2)**2/n1 + (1-r2**2)**2/n2)
    U = r1 - r2 + 1.96 * np.sqrt((1-r1**2)**2/n1 + (1-r2**2)**2/n2)
    
    est = r1 - r2
    se = (U - L)/(2*1.96)
    z = est/se
    p = np.exp(-0.717*z-0.416*z**2)
    print('p =' ,p)
    print('95%')
    return (L,U)