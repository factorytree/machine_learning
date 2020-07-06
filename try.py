import matplotlib.pyplot as plt

x=[1,2,3,4,5,6]
layer3=[0.02575267,0.02164056,0.01334059,0.01167686,0.01109828,0.00975102]
layer7=[0.07275619,0.03361349,0.02751755,0.01862863,0.01529884,0.01366985]
layer30=[0.03436182,0.02421587,0.01820305,0.0129549,0.01180482,0.01169842]
layer40=[0.17860205,0.10375258,0.09441737,0.08451037,0.06730461,0.06094614]

def graph():
    plt.plot(x, layer3,'-o', color='skyblue', label='layer3 of initial model')
    plt.plot(x, layer7,'-o', color='blue', label='layer7 of initial model')
    plt.plot(x, layer30,'-o', color='pink', label='layer30 of improved model')
    plt.plot(x, layer40,'-o', color='red', label='layer40 of improved model')
    plt.legend()

    plt.xlabel('Component')
    plt.ylabel('Contribution Rate')
    plt.show()

graph()