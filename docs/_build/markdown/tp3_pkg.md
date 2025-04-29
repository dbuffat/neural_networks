# tp3_pkg package

## Submodules

## tp3_pkg.neurones module

### *class* tp3_pkg.neurones.NeuralNetwork(lambda_1=1.0, lambda_2=0.1, iterations=100)

Bases : `object`

Implémentation d’un réseau de neurones simple pour la classification avec rétropropagation.

#### lambda_1

Coefficient d’apprentissage initial.

* **Type:**
  float

#### lambda_2

Coefficient d’apprentissage secondaire.

* **Type:**
  float

#### iterations

Nombre d’itérations d’entraînement.

* **Type:**
  int

#### Données

#### data_numpy

Jeu de données chargé depuis un fichier CSV.

* **Type:**
  numpy.ndarray

#### X_train

Données d’entraînement normalisées.

* **Type:**
  numpy.ndarray

#### Y_train

Étiquettes des données d’entraînement.

* **Type:**
  numpy.ndarray

#### X_dev

Données de validation normalisées.

* **Type:**
  numpy.ndarray

#### Y_dev

Étiquettes des données de validation.

* **Type:**
  numpy.ndarray

### Paramètres du réseau

### W0, W1

Matrices de poids pour chaque couche.

* **Type:**
  numpy.ndarray

### b0, b1

Vecteurs de biais pour chaque couche.

* **Type:**
  numpy.ndarray

#### Gradients

### dJdW0, dJdW1

Gradients des poids pour la rétropropagation.

* **Type:**
  numpy.ndarray

### dJdb0, dJdb1

Gradients des biais pour la rétropropagation.

* **Type:**
  numpy.ndarray

#### Autres

#### rep

Liste des prédictions du réseau.

* **Type:**
  list

#### success_

Historique des précisions à chaque étape d’entraînement.

* **Type:**
  list

#### \_\_init_\_(lambda_1=1.0, lambda_2=0.1, iterations=100)

#### RunMain()

Lance l’ensemble du processus : lecture des données, normalisation,
initialisation des paramètres et entraînements (avec deux taux d’entrainement).

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_read_data()

Lit les données d’entraînement depuis un fichier CSV et les convertit en tableaux NumPy.

Cet appel mélange les données pour garantir une distribution aléatoire.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_sampling()

Sépare les données en deux ensembles :
- Échantillon de validation (X_dev, Y_dev)
- Échantillon d’entraînement (X_train, Y_train)

Cela utilise les 1000 premiers exemples pour la validation et le reste pour l’entraînement.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_norm()

Normalise les caractéristiques (X) en divisant chaque valeur par 255
pour s’assurer que toutes les caractéristiques se situent entre 0 et 1.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_init_W_b()

Initialise les poids et biais pour le réseau de neurones avec des valeurs aléatoires
entre -0.5 et 0.5.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_init_derivate()

Initialise les matrices de dérivées à zéro pour les poids et biais
du réseau.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_ReLU(Z)

Applique la fonction d’activation ReLU à un tableau.

* **Paramètres:**
  **Z** (*numpy.ndarray*) – Entrée pour la fonction d’activation.
* **Renvoie:**
  Sortie après application de ReLU.
* **Type renvoyé:**
  numpy.ndarray

#### \_softmax(Z)

Applique une fonction softmax à l’entrée.

* **Paramètres:**
  **Z** (*numpy.ndarray*) – Vecteur contenant les valeurs d’entrée.
* **Renvoie:**
  Vecteur après application de softmax.
* **Type renvoyé:**
  numpy.ndarray

#### \_prop_av(X_i)

Effectue une propagation en avant à travers le réseau pour une seule entrée.
Le vecteur A1 est le vecteur de réponse, l’indice qui a la plus grande valeur associée est considérée comme la réponse du réseau de neurones.

* **Paramètres:**
  **X_i** (*numpy.ndarray*) – Exemples d’entrée pour le réseau.
* **Type renvoyé:**
  None

#### \_dReLU_dz(Z)

Calcule la dérivée de ReLU par rapport à Z en retournant 1 pour les
valeurs positives, et 0 pour les valeurs négatives.

* **Paramètres:**
  **Z** (*numpy.ndarray*) – Entrée pour laquelle calculer la dérivée.
* **Renvoie:**
  Dérivée de ReLU évaluée point par point.
* **Type renvoyé:**
  numpy.ndarray

#### \_value_to_array(Y_i)

Convertit une valeur Y en tableau encodé unilatéral.

* **Paramètres:**
  **Y_i** (*int*) – Valeur cible à encoder.
* **Renvoie:**
  Tableau de longueur 10 avec un seul « 1 » au bon indice.
* **Type renvoyé:**
  numpy.ndarray

#### \_prop_ar(X_i, Y_i)

Effectue une rétropropagation pour calculer les gradients des poids et biais
en utilisant les erreurs entre la sortie prédite et la cible.

* **Paramètres:**
  * **X_i** (*numpy.ndarray*) – Exemple d’entrée utilisé pour la rétropropagation.
  * **Y_i** (*int*) – Étiquette de classe réelle pour l’entrée X_i.
* **Type renvoyé:**
  None

#### \_refresh_matrix(lambda_)

Met à jour les poids et les biais avec les gradients calculés via rétropropagation,
appliquant la descente de gradient stochastique.

* **Paramètres:**
  **lambda** (*float*) – Coefficient d’apprentissage utilisé pour ajuster les paramètres.
* **Type renvoyé:**
  None

#### \_end_value()

Détermine la classe prédite en prenant l’indice de la valeur maximale du vecteur de sortie.

Une fois déterminée, la classe prédite est ajoutée à la liste des réponses du réseau.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_comparaison(use)

Compare les prédictions du réseau avec les véritables étiquettes pour calculer la précision.

* **Paramètres:**
  **use** (*str*) – Indique quel ensemble de données utiliser : “train” pour l’entraînement
  ou tout autre valeur pour la validation.
* **Renvoie:**
  Le pourcentage de détections correctes.
* **Type renvoyé:**
  float

#### \_run_train(lambda_)

Entraîne le réseau de neurones sur l’ensemble des données d’entraînement
en utilisant la rétropropagation.

* **Paramètres:**
  **lambda** (*float*) – Coefficient d’apprentissage utilisé pour la mise à jour des poids
  et des biais.
* **Type renvoyé:**
  None

#### \_run_dev()

Effectue la validation du réseau de neurones sur l’ensemble des données de validation.

Cette méthode applique une propagation avant pour chaque exemple, compare les
prédictions générées avec les étiquettes réelles et affiche la précision.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_\_dict_\_ *= mappingproxy({'_\_module_\_': 'tp3_pkg.neurones', '_\_doc_\_': "\\n    Implémentation d'un réseau de neurones simple pour la classification avec rétropropagation.\\n\\n    Attributes\\n    ----------\\n    lambda_1 : float\\n        Coefficient d'apprentissage initial.\\n    lambda_2 : float\\n        Coefficient d'apprentissage secondaire.\\n    iterations : int\\n        Nombre d'itérations d'entraînement.\\n    \\n    Données :\\n    data_numpy : numpy.ndarray\\n        Jeu de données chargé depuis un fichier CSV.\\n    X_train : numpy.ndarray\\n        Données d'entraînement normalisées.\\n    Y_train : numpy.ndarray\\n        Étiquettes des données d'entraînement.\\n    X_dev : numpy.ndarray\\n        Données de validation normalisées.\\n    Y_dev : numpy.ndarray\\n        Étiquettes des données de validation.\\n\\n    Paramètres du réseau :\\n    W0, W1 : numpy.ndarray\\n        Matrices de poids pour chaque couche.\\n    b0, b1 : numpy.ndarray\\n        Vecteurs de biais pour chaque couche.\\n    \\n    Gradients :\\n    dJdW0, dJdW1 : numpy.ndarray\\n        Gradients des poids pour la rétropropagation.\\n    dJdb0, dJdb1 : numpy.ndarray\\n        Gradients des biais pour la rétropropagation.\\n\\n    Autres :\\n    rep : list\\n        Liste des prédictions du réseau.\\n    success_ : list\\n        Historique des précisions à chaque étape d'entraînement.\\n    ", '_\_init_\_': <function NeuralNetwork._\_init_\_>, 'RunMain': <function NeuralNetwork.RunMain>, '_read_data': <function NeuralNetwork._read_data>, '_sampling': <function NeuralNetwork._sampling>, '_norm': <function NeuralNetwork._norm>, '_init_W_b': <function NeuralNetwork._init_W_b>, '_init_derivate': <function NeuralNetwork._init_derivate>, '_ReLU': <function NeuralNetwork._ReLU>, '_softmax': <function NeuralNetwork._softmax>, '_prop_av': <function NeuralNetwork._prop_av>, '_dReLU_dz': <function NeuralNetwork._dReLU_dz>, '_value_to_array': <function NeuralNetwork._value_to_array>, '_prop_ar': <function NeuralNetwork._prop_ar>, '_refresh_matrix': <function NeuralNetwork._refresh_matrix>, '_end_value': <function NeuralNetwork._end_value>, '_comparaison': <function NeuralNetwork._comparaison>, '_run_train': <function NeuralNetwork._run_train>, '_run_dev': <function NeuralNetwork._run_dev>, '_\_dict_\_': <attribute '_\_dict_\_' of 'NeuralNetwork' objects>, '_\_weakref_\_': <attribute '_\_weakref_\_' of 'NeuralNetwork' objects>, '_\_annotations_\_': {}})*

#### \_\_module_\_ *= 'tp3_pkg.neurones'*

#### \_\_weakref_\_

list of weak references to the object (if defined)

## tp3_pkg.neurones_keras module

### *class* tp3_pkg.neurones_keras.NeuralNetworkKeras

Bases : `object`

Classe pour entraîner et évaluer des réseaux de neurones sur le jeu de données MNIST
en utilisant Keras.

#### X_train

Données d’entraînement normalisées.

* **Type:**
  numpy.ndarray

#### Y_train

Étiquettes des données d’entraînement.

* **Type:**
  numpy.ndarray

#### X_test

Données de test normalisées.

* **Type:**
  numpy.ndarray

#### Y_test

Étiquettes des données de test.

* **Type:**
  numpy.ndarray

#### Y_train_arr

Étiquettes d’entraînement en one-hot encoding.

* **Type:**
  numpy.ndarray

#### Y_test_arr

Étiquettes de test en one-hot encoding.

* **Type:**
  numpy.ndarray

#### model

Modèle de réseau de neurones.

* **Type:**
  keras.models.Sequential

#### Adam

Optimiseur Adam.

* **Type:**
  keras.optimizers.Adam

#### out

Historique de l’entraînement du modèle.

* **Type:**
  keras.callbacks.History

#### \_\_init_\_()

#### \_compile_data()

Compile les données d’entraînement et de test en important et en convertissant
les valeurs des étiquettes en one-hot encoding.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### RunMain()

Lance les différents modèles les uns à la suite des autres.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### Run1()

Lance le premier modèle d’entrainement du réseau de neurones.
L’intérêt est de pouvoir choisir quel modèle on veut utiliser. Cela permet d’éviter de sur-utiliser la RAM.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### Run2()

Lance le second modèle d’entrainement du réseau de neurones.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### Run3()

Lance le troisième modèle d’entrainement du réseau de neurones.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### Run4()

Lance le quatrième modèle d’entrainement du réseau de neurones.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### Run5()

Lance le cinquième modèle d’entrainement du réseau de neurones.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_import_data()

Importe et normalise le jeu de données MNIST.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_value_to_array()

Convertit les étiquettes en one-hot encoding pour l’entraînement et le test.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_train_1(n=300)

Entraîne un modèle de réseau de neurones simple avec une couche dense.

* **Paramètres:**
  **n** (*int* *,* *optional*) – Nombre d’époques pour l’entraînement, par défaut 300.
* **Type renvoyé:**
  None

#### \_train_2()

Entraîne un modèle de réseau de neurones avec un taux d’apprentissage différent.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_train_3()

Entraîne un modèle de réseau de neurones avec une architecture étendue.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_train_4()

Entraîne un modèle avec deux couches cachées étendues.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### \_train_5()

Entraîne un modèle similaire au quatrième mais avec un lot de données plus petit.

* **Paramètres:**
  **None**
* **Type renvoyé:**
  None

#### PlotOut(tag)

Trace les courbes d’apprentissage et de validation pour le modèle.

* **Paramètres:**
  **tag** (*str*) – Titre pour le graphique.
* **Type renvoyé:**
  None

#### \_\_dict_\_ *= mappingproxy({'_\_module_\_': 'tp3_pkg.neurones_keras', '_\_doc_\_': "\\n    Classe pour entraîner et évaluer des réseaux de neurones sur le jeu de données MNIST \\n    en utilisant Keras.\\n\\n    Attributes\\n    ----------\\n    X_train : numpy.ndarray\\n        Données d'entraînement normalisées.\\n    Y_train : numpy.ndarray\\n        Étiquettes des données d'entraînement.\\n    X_test : numpy.ndarray\\n        Données de test normalisées.\\n    Y_test : numpy.ndarray\\n        Étiquettes des données de test.\\n    Y_train_arr : numpy.ndarray\\n        Étiquettes d'entraînement en one-hot encoding.\\n    Y_test_arr : numpy.ndarray\\n        Étiquettes de test en one-hot encoding.\\n    model : keras.models.Sequential\\n        Modèle de réseau de neurones.\\n    Adam : keras.optimizers.Adam\\n        Optimiseur Adam.\\n    out : keras.callbacks.History\\n        Historique de l'entraînement du modèle.\\n    ", '_\_init_\_': <function NeuralNetworkKeras._\_init_\_>, '_compile_data': <function NeuralNetworkKeras._compile_data>, 'RunMain': <function NeuralNetworkKeras.RunMain>, 'Run1': <function NeuralNetworkKeras.Run1>, 'Run2': <function NeuralNetworkKeras.Run2>, 'Run3': <function NeuralNetworkKeras.Run3>, 'Run4': <function NeuralNetworkKeras.Run4>, 'Run5': <function NeuralNetworkKeras.Run5>, '_import_data': <function NeuralNetworkKeras._import_data>, '_value_to_array': <function NeuralNetworkKeras._value_to_array>, '_train_1': <function NeuralNetworkKeras._train_1>, '_train_2': <function NeuralNetworkKeras._train_2>, '_train_3': <function NeuralNetworkKeras._train_3>, '_train_4': <function NeuralNetworkKeras._train_4>, '_train_5': <function NeuralNetworkKeras._train_5>, 'PlotOut': <function NeuralNetworkKeras.PlotOut>, '_\_dict_\_': <attribute '_\_dict_\_' of 'NeuralNetworkKeras' objects>, '_\_weakref_\_': <attribute '_\_weakref_\_' of 'NeuralNetworkKeras' objects>, '_\_annotations_\_': {}})*

#### \_\_module_\_ *= 'tp3_pkg.neurones_keras'*

#### \_\_weakref_\_

list of weak references to the object (if defined)

## Module contents
