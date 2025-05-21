# test_dataset_with_save.py

import os
import sys
sys.path.insert(0, '../helper')
import dataset
import numpy as np

def main():
    # Numero di feature
    N = 10

    # Cartella dove salvare i file
    output_folder = "output_dataset"
    dataset_name = "random"

    # Istanzia la classe Dataset
    ds = dataset.Dataset()
    ds.N = N  # necessario per alcuni metodi

    # Crea un dataset random
    training, label = ds.createTrainingRandom(N)

    # Mostra un esempio
    print("Esempio di etichette:", label)
    print("Primo vettore di training:", training[0])
    print("Forma del dataset:", training.shape)

    # Salva il dataset su disco
    print(f"Salvataggio in: {output_folder}")
    #ds.saveDataset(output_folder, dataset_name, training, label)
    ds.createAndSaveDataset(_folder=output_folder, _dataset=dataset_name, _label=None, _nr=2)

    print("Salvataggio completato.")

if __name__ == "__main__":
    main()
