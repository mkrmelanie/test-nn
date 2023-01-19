import numpy
import matplotlib

infoArray = numpy.array(numpy.random.randint(0, high=2, size=(3,5)), dtype=int)

Vi = int()
Vj = int()

while Vi != Vj:
    Vi = infoArray[(numpy.random.randint(0, high=2, dtype=int)), 0]
    Vj = infoArray[(numpy.random.randint(0, high=2, dtype=int)), 0]

def infoStorage():
    numpy.sum((2*Vi-1)*(2*Vj-1))

Tij = infoStorage()
Ui = 0

def infoUpdate():
    if numpy.sum(Tij*Vj) > Ui:
        return 1
    else:
        return 0