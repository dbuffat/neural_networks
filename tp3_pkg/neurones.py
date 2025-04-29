import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from progress.bar import Bar
import time

version = "1.0"

class NeuralNetwork:
    """
    Implémentation d'un réseau de neurones simple pour la classification avec rétropropagation.

    Attributes
    ----------
    Variables d'entrainement :
        lambda_1 : float
            Coefficient d'apprentissage initial.
        lambda_2 : float
            Coefficient d'apprentissage secondaire.
        iterations : int
            Nombre d'itérations d'entraînement.
        
    Données :
        data_numpy : numpy.ndarray
            Jeu de données chargé depuis un fichier CSV.
        X_train : numpy.ndarray
            Données d'entraînement normalisées.
        Y_train : numpy.ndarray
            Étiquettes des données d'entraînement.
        X_dev : numpy.ndarray
            Données de validation normalisées.
        Y_dev : numpy.ndarray
            Étiquettes des données de validation.

    Paramètres du réseau :
        W0, W1 : numpy.ndarray
            Matrices de poids pour chaque couche.
        b0, b1 : numpy.ndarray
            Vecteurs de biais pour chaque couche.
        
    Gradients :
        dJdW0, dJdW1 : numpy.ndarray
            Gradients des poids pour la rétropropagation.
        dJdb0, dJdb1 : numpy.ndarray
            Gradients des biais pour la rétropropagation.

    Autres :
        rep : list
            Liste des prédictions du réseau.
        success_ : list
            Historique des précisions à chaque étape d'entraînement.
    """
    def __init__(self, lambda_1=1.0, lambda_2=0.1, iterations=100):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.iterations = iterations

        self.data_numpy = None
        self.X_dev = None
        self.Y_dev = None
        self.X_train = None
        self.Y_train = None
        
        self.W0 = None
        self.W1 = None
        self.b0 = None
        self.b1 = None
        
        self.A0 = None
        self.A1 = None
        self.Z0 = None
        self.Z1 = None
        
        self.dJdW0 = None
        self.dJdW1 = None
        self.dJdb0 = None
        self.dJdb1 = None

        self.rep = []
        self.success_ = []

    def RunMain(self):
        """
        Lance l'ensemble du processus : lecture des données, normalisation,
        initialisation des paramètres et entraînements (avec deux taux d'entrainement).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        start = time.time()
        
        self._read_data()
        self._sampling()
        self._norm()
        self._init_W_b()

        self._run_train(self.lambda_1)
        self._run_train(self.lambda_2)

        self._run_dev()
        
        stop = time.time()
        self._plot_success()
        np.savez(file = "weight_and_biases.npz", W_0 = self.W0, b_0 = self.b0, W_1 = self.W1, b_1 = self.b1)
        duration = stop - start
        print(f"La durée de calcul est de {duration:.2f} secondes")

    def _read_data(self):
        """
        Lit les données d'entraînement depuis un fichier CSV et les convertit en tableaux NumPy.

        Cet appel mélange les données pour garantir une distribution aléatoire.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        df = pd.read_csv('tp3_pkg/train.csv')
        self.data_numpy = df.to_numpy()
        np.random.shuffle(self.data_numpy)

    def _sampling(self):
        """
        Sépare les données en deux ensembles :
        - Échantillon de validation (X_dev, Y_dev)
        - Échantillon d'entraînement (X_train, Y_train)

        Cela utilise les 1000 premiers exemples pour la validation et le reste pour l'entraînement.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Échantillon de validation
        self.X_dev = self.data_numpy[:1000, 1:]
        self.Y_dev = self.data_numpy[:1000, 0]

        # Échantillon d'entraînement
        self.X_train = self.data_numpy[1000:, 1:]
        self.Y_train = self.data_numpy[1000:, 0]

        # Liberation de la memoire
        self.data_numpy = None

    def _norm(self):
        """
        Normalise les caractéristiques (X) en divisant chaque valeur par 255
        pour s'assurer que toutes les caractéristiques se situent entre 0 et 1.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        #norm = np.max(self.data_numpy[:, 1:])
        self.X_train = self.X_train / 255.
        self.X_dev = self.X_dev / 255.

    def _init_W_b(self):
        """
        Initialise les poids et biais pour le réseau de neurones avec des valeurs aléatoires
        entre -0.5 et 0.5.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Initialisation des poids
        self.W0 = np.random.uniform(-0.5, 0.5, (784, 10))
        self.W1 = np.random.uniform(-0.5, 0.5, (10, 10))

        # Initialisation des biais
        self.b0 = np.random.uniform(-0.5, 0.5, 10)
        self.b1 = np.random.uniform(-0.5, 0.5, 10)

    def _init_derivate(self):
        """
        Initialise les matrices de dérivées à zéro pour les poids et biais
        du réseau.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.dJdW0 = np.zeros((784,10))
        self.dJdW1 = np.zeros((10,10))
        self.dJdb0 = np.zeros(10)
        self.dJdb1 = np.zeros(10)

    def _ReLU(self, Z):
        """
        Applique la fonction d'activation ReLU à un tableau.

        Parameters
        ----------
        Z : numpy.ndarray
            Entrée pour la fonction d'activation.

        Returns
        -------
        numpy.ndarray
            Sortie après application de ReLU.
        """
        return np.maximum(Z, 0)

    def _softmax(self, Z):
        """
        Applique une fonction softmax à l'entrée.

        Parameters
        ----------
        Z : numpy.ndarray
            Vecteur contenant les valeurs d'entrée.

        Returns
        -------
        numpy.ndarray
            Vecteur après application de softmax.
        """
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z)

    def _prop_av(self, X_i):
        """
        Effectue une propagation en avant à travers le réseau pour une seule entrée.
        Le vecteur A1 est le vecteur de réponse, l'indice qui a la plus grande valeur associée est considérée comme la réponse du réseau de neurones.

        Parameters
        ----------
        X_i : numpy.ndarray
            Exemples d'entrée pour le réseau.

        Returns
        -------
        None
        """
        self.Z0 = self.W0.T @ X_i + self.b0
        self.A0 = self._ReLU(self.Z0)
        self.Z1 = self.W1.T @ self.A0 + self.b1
        self.A1 = self._softmax(self.Z1)

        self._end_value()

    def _dReLU_dz(self, Z):
        """
        Calcule la dérivée de ReLU par rapport à Z en retournant 1 pour les
        valeurs positives, et 0 pour les valeurs négatives.

        Parameters
        ----------
        Z : numpy.ndarray
            Entrée pour laquelle calculer la dérivée.

        Returns
        -------
        numpy.ndarray
            Dérivée de ReLU évaluée point par point.
        """
        return Z > 0

    def _value_to_array(self, Y_i):
        """
        Convertit une valeur Y en tableau encodé unilatéral.

        Parameters
        ----------
        Y_i : int
            Valeur cible à encoder.

        Returns
        -------
        numpy.ndarray
            Tableau de longueur 10 avec un seul "1" au bon indice.
        """
        table = np.zeros(10)
        table[Y_i] = 1
        return table

    def _prop_ar(self, X_i, Y_i):
        """
        Effectue une rétropropagation pour calculer les gradients des poids et biais
        en utilisant les erreurs entre la sortie prédite et la cible.

        Parameters
        ----------
        X_i : numpy.ndarray
            Exemple d'entrée utilisé pour la rétropropagation.
        Y_i : int
            Étiquette de classe réelle pour l'entrée X_i.

        Returns
        -------
        None
        """
        delta_1 = self.A1 - self._value_to_array(Y_i)
        delta_0 = self._dReLU_dz(self.Z0) * (self.W1 @ delta_1)

        self.dJdb0 += delta_0
        self.dJdb1 += delta_1
        
        self.dJdW0 += np.outer(X_i, delta_0)        
        self.dJdW1 += np.outer(self.A0, delta_1)

    def _refresh_matrix(self, lambda_):
        """
        Met à jour les poids et les biais avec les gradients calculés via rétropropagation,
        appliquant la descente de gradient stochastique.

        Parameters
        ----------
        lambda_ : float
            Coefficient d'apprentissage utilisé pour ajuster les paramètres.

        Returns
        -------
        None
        """
        D = self.X_train.shape[0]
        self.W0 = self.W0 - lambda_/D * self.dJdW0 # DIM finale (784,10)
        self.W1 = self.W1 - lambda_/D * self.dJdW1 # DIM finale (10,10)

        self.b0 = self.b0 - lambda_/D * self.dJdb0 # DIM finale (10,1)
        self.b1 = self.b1 - lambda_/D * self.dJdb1 # DIM finale (10,1)

    def _end_value(self):
        """
        Détermine la classe prédite en prenant l'indice de la valeur maximale du vecteur de sortie.

        Une fois déterminée, la classe prédite est ajoutée à la liste des réponses du réseau.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        idx = np.argmax(self.A1)
        #rep = np.zeros(10)
        #rep[idx] = 1
        self.rep.append(idx)

    def _comparaison(self, use):
        """
        Compare les prédictions du réseau avec les véritables étiquettes pour calculer la précision.

        Parameters
        ----------
        use : str
            Indique quel ensemble de données utiliser : 'train' pour l'entraînement
            ou tout autre valeur pour la validation.

        Returns
        -------
        float
            Le pourcentage de détections correctes.
        """
        rep = np.array(self.rep)
        
        if use == 'train':
            validation = self.Y_train
        else:
            validation = self.Y_dev

        success = np.mean(rep == validation) * 100
        print(f"Détéction correcte : {success:.2f} %")
        self.success_.append(success)
        return success

    def _run_train(self, lambda_):
        """
        Entraîne le réseau de neurones sur l'ensemble des données d'entraînement
        en utilisant la rétropropagation.

        Parameters
        ----------
        lambda_ : float
            Coefficient d'apprentissage utilisé pour la mise à jour des poids
            et des biais.

        Returns
        -------
        None
        """
        for i in range(self.iterations):
            self._init_derivate()

            for j in range(self.X_train.shape[0]):
                self._prop_av(self.X_train[j,:])
                self._prop_ar(self.X_train[j,:], self.Y_train[j])
                
            self._refresh_matrix(lambda_)
            
            if i % 10 == 0 :
                print( '\n' + '========== Entrainement ==========')
                self._comparaison('train')
                print('==================================')
            
            self.rep = []

        print(f"b0 a la derniere iteration :\n{self.b0}")
        print(f"\nW0 a la derniere iteration :\n{self.W0}")
        print(f"\nb1 a la derniere iteration :\n{self.b1}")
        print(f"\nW1 a la derniere iteration :\n{self.W1}")

    def _run_dev(self):
        """
        Effectue la validation du réseau de neurones sur l'ensemble des données de validation.

        Cette méthode applique une propagation avant pour chaque exemple, compare les
        prédictions générées avec les étiquettes réelles et affiche la précision.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in range(self.X_dev.shape[0]):
            self._prop_av(self.X_dev[i,:])
                            
        print( '\n' + '========== Validation ==========')
        self._comparaison('dev')
        print('================================')
            
        self.rep = []

    def _plot_success(self):
        """
            Trace le taux de succès en fonction des itérations d'entraînement.

            Parameters
            ----------
            None

            Returns
            -------
            None
            """
        if not self.success_:
            print("Aucun taux de succès enregistré. Veuillez entraîner le modèle d'abord.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(range(0, len(self.success_) * 10, 10), self.success_, marker='o', linestyle='-', color='blue')
        plt.title("Taux de succès en fonction des itérations d'entraînement")
        plt.xlabel("Itérations")
        plt.ylabel("Taux de succès (%)")
        plt.grid(True)
        plt.show()
