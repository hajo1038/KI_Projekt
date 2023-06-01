from Neuron import Neuron
from Model import Model

def main():
    # Use a breakpoint in the code line below to debug your script.
    model = Model()  # Press ⌘F8 to toggle the breakpoint.
    model.add_layer(10)
    model.add_layer(5)
    model.add_layer(2)
    print("Test")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# Layer mit Bias und Neuronen. Muss Informationen über vorherige und nachfolgende Schichten erhalten.
# Aktivierungsfunktion?

# Neuronen mit Gewichten
