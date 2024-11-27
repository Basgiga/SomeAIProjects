import numpy as np

CO_ILE_PREZENTACJA = 1
ILE_EPOK = 1000
LEARNING_RATE = 0.1
CZY_PREZENTACJA = 0
CZY_PREZENTACJA_NEURO = 0

def unipolarny(wart):
    #print(wart)
    if wart >=0:
        return 1
    return 0

def bipolarny(wart):
    if wart >=0:
        return 1
    return -1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Perceptron:
    def __init__(self, ile_wejsc, f_aktyw, learning_rate = LEARNING_RATE):
        self.wagi = np.random.uniform(-1, 1, ile_wejsc)
        self.bias = np.random.uniform(-1, 1)
        self.funkcja_aktywacji = f_aktyw
        self.learning_rate = learning_rate

    def prezentacja(self, ypred, error,dana,ytrue):
        print(f"------------PREZENTACJA PERCEPTRONU-------------\n")
        print(f"dla danej:{dana}\nlabel:{ytrue}\n")
        print(f"predykcja:{ypred}\nblad:{error}\n")
        print(f"wagi:{self.wagi}\nbias:{self.bias}\n")

    def predykcja(self, wejscie):
        return self.funkcja_aktywacji(np.dot(self.wagi, wejscie) + self.bias)

    def trening(self, dane, labele, epochs = ILE_EPOK):
        for kt in range(epochs):
            for i in range(len(dane)):
                dana = dane[i]
                ypred = self.predykcja(dana)
                ytrue = labele[i]
                error = ytrue - ypred
                self.wagi = self.wagi + self.learning_rate * error * dane[i]
                self.bias = self.bias + self.learning_rate * error
                if kt%CO_ILE_PREZENTACJA == 0:
                    if(CZY_PREZENTACJA == 1):
                        print(f"########### Epoka: {kt} ############\n")
                        self.prezentacja(ypred,error,dana,ytrue)
                        print(f"\n")



def LUB_uni():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_uni = np.array([0,1,1,1])     # unipolkarne
    tr_labele_bi = np.array([-1,1,1,1])     # bipoarne

    # LUB, 2 wejscia, unipolarny
    perceptronLU = Perceptron(ile_wejsc=2, f_aktyw=unipolarny)
    perceptronLU.trening(tr_dane,tr_labele_uni)

    # testowanie
    print("\nPerceptron dla LUB z 2ma wejsciami unipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronLU.predykcja(wejscia)}")

def LUB_bi():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_bi = np.array([-1,1,1,1])     # bipoarne

    # LUB, 2 wejscia, bipolarny
    perceptronLB = Perceptron(ile_wejsc=2, f_aktyw=bipolarny)
    perceptronLB.trening(tr_dane,tr_labele_bi)

    # testowanie
    print("\nPerceptron dla LUB z 2ma wejsciami bipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronLB.predykcja(wejscia)}")

def AND_uni():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_uni = np.array([0,0,0,1])     # unipolkarne
    tr_labele_bi = np.array([-1,-1,-1,1])     # bipoarne

    # I, 2 wejscia, unipolarny
    perceptronAU = Perceptron(ile_wejsc=2, f_aktyw=unipolarny)
    perceptronAU.trening(tr_dane,tr_labele_uni)

    # testowanie
    print("\nPerceptron dla I z 2ma wejsciami unipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronAU.predykcja(wejscia)}")


def AND_bi():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_uni = np.array([0,0,0,1])     # unipolkarne
    tr_labele_bi = np.array([-1,-1,-1,1])     # bipoarne

    # I, 2 wejscia, unipolarny
    perceptronAB = Perceptron(ile_wejsc=2, f_aktyw=bipolarny)
    perceptronAB.trening(tr_dane,tr_labele_bi)

    # testowanie
    print("\nPerceptron dla I z 2ma wejsciami bipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronAB.predykcja(wejscia)}")

def NOT_uni():
    tr_dane = np.array([0,1])
    tr_labele_uni = np.array([1,0])     # unipolkarne
    tr_labele_bi = np.array([1,-1])     # bipoarne

    # NOT, 2 wejscia, unipolarny
    perceptronNU = Perceptron(ile_wejsc=1, f_aktyw=unipolarny)
    perceptronNU.trening(tr_dane,tr_labele_uni)

    # testowanie
    print("\nPerceptron dla NIE z 1 wejsciem unipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronNU.predykcja(wejscia)}")

def NOT_bi():
    tr_dane = np.array([0,1])
    tr_labele_uni = np.array([1,0])     # unipolkarne
    tr_labele_bi = np.array([1,-1])     # bipoarne

    # NOT, 2 wejscia, unipolarny
    perceptronNU = Perceptron(ile_wejsc=1, f_aktyw=bipolarny)
    perceptronNU.trening(tr_dane,tr_labele_bi)

    # testowanie
    print("\nPerceptron dla NIE z 1 wejsciem unipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronNU.predykcja(wejscia)}")


def ALBO_uni():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_uni = np.array([0,1,1,0])     # unipolkarne
    tr_labele_bi = np.array([-1,1,1,-1])     # bipoarne

    # ALBO, 2 wejscia, unipolarny
    perceptronALU = Perceptron(ile_wejsc=2, f_aktyw=unipolarny)
    perceptronALU.trening(tr_dane,tr_labele_uni)

    # testowanie
    print("\nPerceptron dla ALBO z 2ma wejsciami unipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronALU.predykcja(wejscia)}")

def ALBO_bi():
    tr_dane = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_uni = np.array([0,1,1,0])     # unipolkarne
    tr_labele_bi = np.array([-1,1,1,-1])     # bipoarne

    # ALBO, 2 wejscia, unipolarny
    perceptronALB = Perceptron(ile_wejsc=2, f_aktyw=bipolarny)
    perceptronALB.trening(tr_dane,tr_labele_bi)

    # testowanie
    print("\nPerceptron dla ALBO z 2ma wejsciami bipolarny\n")
    for wejscia in tr_dane:
        print(f"wejscie: {wejscia}, Predykcja: {perceptronALB.predykcja(wejscia)}")


class SIECXOR:
    def __init__(self):
        self.ukr1 = Perceptron(ile_wejsc=2, f_aktyw=sigmoid)
        self.ukr2 = Perceptron(ile_wejsc=2, f_aktyw=sigmoid)
        self.wyjscie = Perceptron(ile_wejsc=2, f_aktyw=sigmoid)

    def predykcja(self,wejscie):
        neu1 = self.ukr1.predykcja(wejscie)
        neu2 = self.ukr2.predykcja(wejscie)
        return self.wyjscie.predykcja(np.array([neu1,neu2]))

    def prezentacja(self, ypred, error,dana,ytrue):
        print(f"------------PREZENTACJA SIECI NEURONOWEJ 3 PERCEPTRONOW-------------\n")
        print(f"dla danej:{dana}\nlabel:{ytrue}\n")
        print(f"predykcja:{ypred}\nblad:{error}\n")
        print(f"DLA UKRYTEGO PERCEPTRONU 1:\n wagi :{self.ukr1.wagi}\nbias:{self.ukr1.bias}\n")
        print(f"DLA UKRYTEGO PERCEPTRONU 2:\n wagi :{self.ukr2.wagi}\nbias:{self.ukr2.bias}\n")
        print(f"DLA WYJSCIOWEGO PERCEPTRONU:\n wagi :{self.wyjscie.wagi}\nbias:{self.wyjscie.bias}\n")

    def trening(self, dane, labele, epochs=ILE_EPOK):
        for kt in range(epochs):
            for i in range(len(dane)):
                dana = dane[i]
                ytrue = labele[i]

                neu1 = self.ukr1.predykcja(dana)
                neu2 = self.ukr2.predykcja(dana)
                ypred = self.wyjscie.predykcja(np.array([neu1, neu2]))

                error = ytrue - ypred

                grad_wyjscie = error * ypred * (1 - ypred)
                self.wyjscie.wagi += self.wyjscie.learning_rate * grad_wyjscie * np.array([neu1, neu2])
                self.wyjscie.bias += self.wyjscie.learning_rate * grad_wyjscie

                grad_ukr1 = grad_wyjscie * self.wyjscie.wagi[0] * neu1 * (1 - neu1)
                self.ukr1.wagi += self.ukr1.learning_rate * grad_ukr1 * dana
                self.ukr1.bias += self.ukr1.learning_rate * grad_ukr1

                grad_ukr2 = grad_wyjscie * self.wyjscie.wagi[1] * neu2 * (1 - neu2)
                self.ukr2.wagi += self.ukr2.learning_rate * grad_ukr2 * dana
                self.ukr2.bias += self.ukr2.learning_rate * grad_ukr2

                if kt % CO_ILE_PREZENTACJA == 0:
                    if CZY_PREZENTACJA_NEURO == 1:
                        print(f"########### Epoka: {kt} ############\n")
                        self.prezentacja(ypred, error, dana, ytrue)
                        print(f"\n")

def ALBO_NEURO_uni():
    tr_danea = np.array([[0,0],[0,1],[1,0],[1,1]])
    tr_labele_unia = np.array([0,1,1,0])     # unipolkarne

    ALBOsiec = SIECXOR()
    ALBOsiec.trening(tr_danea,tr_labele_unia)

    # testowanie
    print("\nSiec Neuronowa z 2 warstw (3 neuronow) dla ALBO (unipolarny)\n")
    for wejscia in tr_danea:
        print(f"wejscie: {wejscia}, Predykcja: {ALBOsiec.predykcja(wejscia)}")


print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\nLEARNING RATE = {LEARNING_RATE}, ILE EPOK = {ILE_EPOK}\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
def Zadanie_1_2():
    LUB_uni()
    LUB_bi()
    AND_uni()
    AND_bi()
    NOT_uni()
    NOT_bi()
    ALBO_uni()
    ALBO_bi()

def Zadanie4():
    ALBO_NEURO_uni()


#Zadanie_1_2()
Zadanie4()