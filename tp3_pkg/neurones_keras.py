from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

version = "1.0"

class NeuralNetworkKeras:
    """
    Classe pour entraîner et évaluer des réseaux de neurones sur le jeu de données MNIST 
    en utilisant Keras.

    Attributes
    ----------
        X_train : numpy.ndarray
            Données d'entraînement normalisées.
        Y_train : numpy.ndarray
            Étiquettes des données d'entraînement.
        X_test : numpy.ndarray
            Données de test normalisées.
        Y_test : numpy.ndarray
            Étiquettes des données de test.
        Y_train_arr : numpy.ndarray
            Étiquettes d'entraînement en one-hot encoding.
        Y_test_arr : numpy.ndarray
            Étiquettes de test en one-hot encoding.
        model : keras.models.Sequential
            Modèle de réseau de neurones.
        Adam : keras.optimizers.Adam
            Optimiseur Adam.
        out : keras.callbacks.History
            Historique de l'entraînement du modèle.
    """
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_train_arr = None
        self.Y_test_arr = None
        
        self.model = None
        self.Adam = None
        self.out = None
        
        self._compile_data()
        
    def _compile_data(self):
        """
        Compile les données d'entraînement et de test en important et en convertissant 
        les valeurs des étiquettes en one-hot encoding.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._import_data()
        self._value_to_array()
        
    def RunMain(self):
        """
        Lance les différents modèles les uns à la suite des autres.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        
        self.Run1()
        self.Run2()
        self.Run3()
        self.Run4()
        self.Run5()
        
    def Run1(self):
        """
        Lance le premier modèle d'entrainement du réseau de neurones.
        L'intérêt est de pouvoir choisir quel modèle on veut utiliser. Cela permet d'éviter de sur-utiliser la RAM.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._train_1()
        self.PlotOut('Train 1 n=300')
        self._train_1(n=500)
        self.PlotOut('Train 1 n=500')
        
    def Run2(self):
        """
        Lance le second modèle d'entrainement du réseau de neurones.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._train_2()
        self.PlotOut('Train 2')
        
    def Run3(self):
        """
        Lance le troisième modèle d'entrainement du réseau de neurones.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._train_3()
        self.PlotOut('Train 3')
    
    def Run4(self):
        """
        Lance le quatrième modèle d'entrainement du réseau de neurones.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._train_4()
        self.PlotOut('Train 4')
        
    def Run5(self):
        """
        Lance le cinquième modèle d'entrainement du réseau de neurones.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._train_5()
        self.PlotOut('Train 5')

    def _import_data(self):
        """
        Importe et normalise le jeu de données MNIST.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = keras.datasets.mnist.load_data()
        self.X_train = self.X_train / 255.
        self.X_test = self.X_test / 255.

        print(self.X_train.shape)
        print(self.Y_train.shape)
        print(self.X_test.shape)
        print(self.Y_test.shape)

        self.X_train = self.X_train.reshape(60000, 784)
        self.X_test = self.X_test.reshape(10000, 784)

        print(self.X_train.shape)
        print(self.X_test.shape)

    def _value_to_array(self):
        """
        Convertit les étiquettes en one-hot encoding pour l'entraînement et le test.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.Y_train_arr = keras.utils.to_categorical(self.Y_train, 10)
        self.Y_test_arr = keras.utils.to_categorical(self.Y_test, 10)

    def _train_1(self, n=300):
        """
        Entraîne un modèle de réseau de neurones simple avec une couche dense.

        Parameters
        ----------
        n : int, optional
            Nombre d'époques pour l'entraînement, par défaut 300.

        Returns
        -------
        None
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(10, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.Adam = keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=self.Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.out = self.model.fit(x=self.X_train, y=self.Y_train_arr, batch_size=self.X_train.shape[0], epochs=n, validation_data=(self.X_test, self.Y_test_arr))

        print(self.out.history.keys())

        success = np.array(self.out.history['accuracy'])
        test = success[:-1] - success[1:]
        
        if test[-1] < 10e-3 :
            test = test[test<10e-3]
            print(f"On converge a 10e-3 pres pour {np.argmax(test)} iterations.")
        else :
            print("On n'a pas converge.")

    def _train_2(self):
        """
        Entraîne un modèle de réseau de neurones avec un taux d'apprentissage différent.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(10, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.Adam = keras.optimizers.Adam(learning_rate=0.2)
        self.model.compile(optimizer=self.Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.out = self.model.fit(x=self.X_train, y=self.Y_train_arr, batch_size=self.X_train.shape[0], epochs=300, validation_data=(self.X_test, self.Y_test_arr))

        print(f"Taux de succes max au 2eme entrainement : {np.max(np.array(self.out.history['accuracy']))}")

    def _train_3(self):
        """
        Entraîne un modèle de réseau de neurones avec une architecture étendue.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(500, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.Adam = keras.optimizers.Adam(learning_rate=0.2)
        self.model.compile(optimizer=self.Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.out = self.model.fit(x=self.X_train, y=self.Y_train_arr, batch_size=self.X_train.shape[0], epochs=300, validation_data=(self.X_test, self.Y_test_arr))

    def _train_4(self):
        """
        Entraîne un modèle avec deux couches cachées étendues.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(500, activation='relu'))
        self.model.add(keras.layers.Dense(700, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.Adam = keras.optimizers.Adam(learning_rate=0.2)
        self.model.compile(optimizer=self.Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.out = self.model.fit(x=self.X_train, y=self.Y_train_arr, batch_size=self.X_train.shape[0], epochs=200, validation_data=(self.X_test, self.Y_test_arr))

        print(np.array(self.out.history['accuracy'])[-1])
        print(np.array(self.out.history['val_accuracy'])[-1])

    def _train_5(self):
        """
        Entraîne un modèle similaire au quatrième mais avec un lot de données plus petit.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Input(shape=(784,)))
        self.model.add(keras.layers.Dense(500, activation='relu'))
        self.model.add(keras.layers.Dense(700, activation='relu'))
        self.model.add(keras.layers.Dense(10, activation='softmax'))
        self.model.summary()
        self.Adam = keras.optimizers.Adam(learning_rate=0.2)
        self.model.compile(optimizer=self.Adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.out = self.model.fit(x=self.X_train, y=self.Y_train_arr, batch_size=self.X_train.shape[0]//10, epochs=200, validation_data=(self.X_test, self.Y_test_arr))

    def PlotOut(self, tag):
        """
        Trace les courbes d'apprentissage et de validation pour le modèle.

        Parameters
        ----------
        tag : str
            Titre pour le graphique.

        Returns
        -------
        None
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        plt.subplots_adjust(hspace=0.3)
        
        axs[0, 0].plot(self.out.history['accuracy'])
        axs[0, 0].set_title("Taux de succes pendant l'entrainement")
        axs[0, 0].set_xlabel('iteration')
        axs[0, 0].set_ylabel('accuracy')
        axs[0, 0].axhline(1.0, color='red', linestyle='--', label='Limite')
        axs[0, 0].legend()
        
        axs[0, 1].plot(self.out.history['loss'])
        axs[0, 1].set_title("Estimation de la fonction de perte pendant l'entrainement")
        axs[0, 1].set_xlabel('iteration')
        axs[0, 1].set_ylabel('loss')
        
        axs[1, 0].plot(self.out.history['val_accuracy'])
        axs[1, 0].set_title("Taux de succes pendant la validation")
        axs[1, 0].set_xlabel('iteration')
        axs[1, 0].set_ylabel('accuracy')
        axs[1, 0].axhline(1.0, color='red', linestyle='--', label='Limite')
        axs[1, 0].legend()
        
        axs[1, 1].plot(self.out.history['val_loss'])
        axs[1, 1].set_title("Estimation de la fonction de perte pendant la validation")
        axs[1, 1].set_xlabel('iteration')
        axs[1, 1].set_ylabel('loss')
        
        plt.suptitle(tag, fontsize=20)
        plt.tight_layout()
        plt.show()
