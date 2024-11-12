"""
Decision Tree used with imperfectly labeled data.

Author : Arthur Hoarau
Date : 05/07/2022
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from lib import ibelief
import numpy as np
import math

class EDT(BaseEstimator, ClassifierMixin):
    """
    EDT for Evidential Decision Tree is used to predict labels when input data
    are imperfectly labeled.
    """

    def __init__(self, min_samples_leaf = 1, criterion = "conflict", lbda = 0.5, rf_max_features = "None", max_depth = 100):
        """
        EDT for Evidential Decision Tree is used to predict labels when input data
        are imperfectly labeled.

        Parameters
        -----
        min_samples_leaf: int
            Minimum number of samples in a leaf
        criterion: string
            Usef criterion for splitting nodes. Use "conflict" for Jousselme distance + inclusion degree, "jousselme" for the Jousselme distance, 
            "euclidian" for euclidian distance, "uncertainty" for nons-pecificity + discord degree.

        Returns
        -----
        The instance of the class.
        """

        if (criterion not in ["euclidian", "conflict", "jousselme", "uncertainty"]):
            raise ValueError("Wrong selected criterion")

        # Used to retrieve the state of the model
        self._fitted = False

        self.root_node = TreeNode()
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.lbda = lbda
        self.rf_max_features = rf_max_features
        self.max_depth = max_depth
    
    def score(self, X, y_true, criterion=3):
        """
        Calculate the accuracy score of the model,
        unsig a specific criterion in "Max Credibility", 
        "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions
        criterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        y_pred = self.predict(X, criterion=criterion)

        # Compare with true labels, and compute accuracy
        return accuracy_score(y_true, y_pred)
    
    def print_tree(self):
        """
        Print the fitted tree.
        """

        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        self.root_node.print_tree()

    def get_max_depth(self):
        """
        Find depth of the tree
        """

        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return(self.root_node.max_depth())

    def mean_samples_leafs(self):
        """
        Find mean samples in leafs
        """

        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        return(np.mean(np.array(self.root_node.mean_samples_leafs())))

    def fit(self, X, y):
        """
        Fit the model according to the training data.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y : ndarray
            Labels array

        Returns
        -----
        self : EDT
            The fitted instance of the class.
        """

        # Check for data integrity
        if X.shape[0] != y.shape[0]:
            if X.shape[0] * (self.nb_classes + 1) == y.shape[0]:
                y = np.reshape(y, (-1, self.nb_classes + 1))
            else:
                raise ValueError("X and y must have the same number of rows")

        # Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1] + 1, 2).is_integer():
            y = np.hstack((np.zeros((y.shape[0],1)), y))
        elif not math.log(y.shape[1], 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        # Save X and y
        self.X_trained = X
        self.y_trained = y

        # Choice of the splitting creterion
        if(self.criterion == "conflict"):
            # D matrix for Jousselme distance
            self.d_matrix = ibelief.Dcalculus(self.y_trained[0].size)
            # Compute conflict with Jousselme distance and inclusion
            self.distances = self._compute_inclusion_distances()

        elif(self.criterion == "jousselme"):
            # D matrix for Jousselme distance
            self.d_matrix = ibelief.Dcalculus(self.y_trained[0].size)
            # Compute Jousselme distances
            self.distances = self._compute_jousselme_distances()
        
        elif(self.criterion == "euclidian"):
            # Compute Jousselme distances
            self.distances = self._compute_euclidian_distances()

        elif(self.criterion == "uncertainty"):
            # Compute Pignistic probabilities
            self.pign_prob, self.elements_size = self._compute_prignistic_prob()
        
        # Save size of the dataset
        self.size = self.X_trained.shape[0]

        # Construction of the tree
        self.root_node = TreeNode()
        self._build_tree(np.array(range(self.size)), self.root_node)

        # The model is now fitted
        self._fitted = True

        return self

    def predict_proba(self, X):
        """
        Predict class by returning pignistic probabilities

        Parameters
        -----
        X : ndarray
            Input array of X's

        Returns
        -----
        The pignistic probabilities for each class.
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        result = np.zeros((X.shape[0], self.y_trained.shape[1]))
        for x in range(X.shape[0]):
            result[x] = self._predict(X[x], self.root_node)

        predictions = ibelief.decisionDST(result.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier has not been fitted yet")

        # Predict output bbas for X
        result = np.zeros((X.shape[0], self.y_trained.shape[1]))
        for x in range(X.shape[0]):
            result[x] = self._predict(X[x], self.root_node)

        # Max Plausibility
        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        # Return predictions or both predictions and bbas
        if return_bba:
            return predictions, result
        else:
            return predictions
    
    def _build_tree(self, indices, root_node):
        """
        Recursive method used to build the fit the decision tree

        Parameters
        -----
        indices : ndarray
            Input array of the indeces (X and y)
        root_node : TreeNode
            Root node, tree will be recursively built
        """

        node_depth = root_node.node_depth + 1

        # Find the best attribute et the treshold for continous values
        A, threshold = self._best_gain(indices)

        if node_depth >= self.max_depth:
            A = None

        if A != None:  
            # Categorical values
            if threshold == None:

                # For each attributes, create a node
                for v in np.unique(self.X_trained[indices,A]):
                    index = np.where(self.X_trained[indices,A] == v)[0]
                    node = TreeNode(attribute=A, attribute_value=v, node_depth=node_depth)
                    self._build_tree(index, node)
                    root_node.leafs.append(node)

            # Numerical values 
            else:
                # Left node
                index = indices[np.where(self.X_trained[indices, A].astype(float) < threshold)[0]]
                node = TreeNode(attribute=A, attribute_value=threshold, continuous_attribute=1, node_depth=node_depth)
                self._build_tree(index, node)
                root_node.leafs.append(node) 
                
                # Right node
                index = indices[np.where(self.X_trained[indices, A].astype(float) >= threshold)[0]]
                node = TreeNode(attribute=A, attribute_value=threshold, continuous_attribute=2, node_depth=node_depth)
                self._build_tree(index, node)
                root_node.leafs.append(node) 
        else:
            # Append a mass if the node is a leaf
            root_node.mass = ibelief.DST(self.y_trained[indices].T, 12).flatten()
            root_node.number_leaf = self.y_trained[indices].shape[0]

    def _best_gain(self, indices):
        
        # Compute Info at root node
        info_root = self._compute_info(indices)

        # Stop if the info at root is equal to 0
        if info_root == 0:
            return None, None

        gains = np.zeros(self.X_trained.shape[1])
        thresholds = []
        
        # Select number of features (Random Forest)
        selected_features = []
        if self.rf_max_features == "sqrt":
            selected_features = np.random.choice(range(self.X_trained.shape[1]), int(math.sqrt(self.X_trained.shape[1])), replace=False)
        else:
            selected_features = range(self.X_trained.shape[1])

        # Interate over each attributes
        for A in selected_features:
            sum = 0
            threshold = None
            flag_float = True

            # Find if the attribute is a Categorical or a Numerical value
            try:
                float(self.X_trained[0,A])
            except ValueError:
                flag_float = False

            # Categorical values
            if flag_float == False:
                # Compute sum of Info
                for v in np.unique(self.X_trained[indices,A]):
                    node = indices[np.where(self.X_trained[indices,A] == v)[0]]
                    sum += (node.shape[0] / indices.shape[0]) * self._compute_info(indices)

                    # Min sample leaf pre-pruning
                    if (node.shape[0] < self.min_samples_leaf):
                        sum = info_root
                        break

            # Numerical values
            else:
                # Find best split
                threshold, sum = self._find_treshoold(indices, info_root, A)

            thresholds.append(threshold)

            # Calculate gain
            gain = info_root - sum
            if (gain) > 0:
                gains[A] = gain
            else:
                gains[A] = 0

        # Null gain pre-pruning
        if np.max(gains) == 0:
            return None, None

        # Return the best attribute and the treshold for numerical attributes
        threshold_arg, = np.where(selected_features == np.argmax(gains))
        return np.argmax(gains), thresholds[threshold_arg[0]]

    def _compute_info(self, indices):
        if indices.shape[0] == 0 or indices.shape[0] == 1:
            return 0

        # Choice of the split criterion
        if self.criterion == 'conflict' or self.criterion == 'jousselme' or self.criterion == "euclidian":
            info = self._compute_distance(indices)
        if self.criterion == 'uncertainty':
            info = self._compute_uncertainty(indices)

        return info
    
    # Jousselme distance
    def _compute_distance(self, indices):
        divisor = indices.shape[0]**2 - indices.shape[0]

        mean_distance = np.sum(self.distances[indices][:,indices]) / divisor        

        return mean_distance   

    def _compute_inclusion_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                d_inc = self._compute_inclusion_degree(self.y_trained[i], self.y_trained[j])
                distances[i,j] = (1 - d_inc) * math.sqrt(np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix), self.y_trained[i]-self.y_trained[j])/2.0)

        return distances

    def _compute_jousselme_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                distances[i,j] = math.sqrt(np.dot(np.dot(self.y_trained[i] - self.y_trained[j], self.d_matrix), self.y_trained[i]-self.y_trained[j])/2.0)

        return distances

    def _compute_euclidian_distances(self):
        size = self.y_trained.shape[0]
        distances = np.zeros((size,size))

        for i in range(size):
            for j in range(size):
                distances[i,j] = math.dist(self.y_trained[i], self.y_trained[j])

        return distances

    def _compute_prignistic_prob(self):
        size = self.y_trained.shape[0]

        pign_prob =  np.zeros((size, self.y_trained.shape[1]))
        elemets_size = np.zeros(self.y_trained.shape[1])

        for k in range(size): 
            betp_atoms = ibelief.decisionDST(self.y_trained[k].T, 4, return_prob=True)[0]
            for i in range(1, self.y_trained.shape[1]):
                for j in range(betp_atoms.shape[0]):
                        if ((2**j) & i) == (2**j):
                            pign_prob[k][i] += betp_atoms[j]

        for i in range(1, self.y_trained.shape[1]):
            elemets_size[i] = math.log2(bin(i).count("1"))

        return pign_prob, elemets_size
    
    def _compute_inclusion_degree(self, m1, m2): 
        m1 = m1[:-1]
        m2 = m2[:-1]
        n1 = np.where(m1 > 0)[0]
        n2 = np.where(m2 > 0)[0]

        # If total ignorance, degree is one
        if n1.shape[0] == 0 or n2.shape[0] == 0:
            return 1

        d_inc_l = 0
        d_inc_r = 0
        
        for X1 in n1:
            for X2 in n2:
                if X1 & X2 == X1:
                    d_inc_l += 1
                if X1 & X2 == X2:
                    d_inc_r += 1

        return (1 / (n1.shape[0] * n2.shape[0])) * max(d_inc_r, d_inc_l)
    

    def _compute_uncertainty(self, indices):

        mass = np.mean(self.y_trained[indices], axis=0)
        lbda = self.lbda

        betp_2 = np.mean(self.pign_prob[indices], axis=0)
        betp_2[betp_2 == 0] = 0.001

        n_mass = mass * self.elements_size
        d_mass = -1 * (mass * np.log2(betp_2))

        return ((1 - lbda) * np.sum(n_mass)) + (lbda * np.sum(d_mass))


    def _find_treshoold(self, indices, info_root, A):
        """
        Returns the best treshold for a split

        Returns
        -----
        treshold : float
            value of the best treshold
        info : float
            sum on child nodes Info
        """

        # Find uniques values for the attribute
        values = np.sort(np.unique(self.X_trained[indices,A]).astype(float))
        
        # Find all possible tresholds
        thresholds = []
        for i in range(values.shape[0] - 1):
            thresholds.append((values[i] + values[i + 1]) / 2)

        if len(thresholds) == 0:
            return values[0], info_root

        infos = np.zeros(len(thresholds))

        # For all tresholds, calculate the info gain
        for v in range(len(thresholds)):
            
            left_node = indices[np.where(self.X_trained[indices,A].astype(float) < thresholds[v])[0]]
            info = (left_node.shape[0] / indices.shape[0]) * self._compute_info(left_node)

            right_node = indices[np.where(self.X_trained[indices,A].astype(float) >= thresholds[v])[0]]
            info += (right_node.shape[0] / indices.shape[0]) * self._compute_info(right_node)

            # Min sample leaf
            if (left_node.shape[0] < self.min_samples_leaf or right_node.shape[0] < self.min_samples_leaf):
                info = info_root

            infos[v] = info

        return thresholds[np.argmin(infos)], infos[np.argmin(infos)]

    def _predict(self, X, root_node):
        """
        predict bbas on the input.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        result : ndarray
            Array of normalized bba
        """

        if type(root_node.mass) is np.ndarray:
            return root_node.mass

        for v in root_node.leafs:
            if v.continuous_attribute == 0 and X[v.attribute] == v.attribute_value:
                return self._predict(X, v)
            elif v.continuous_attribute == 1 and X[v.attribute].astype(float) < v.attribute_value:
                return self._predict(X, v)
            elif v.continuous_attribute == 2 and X[v.attribute].astype(float) >= v.attribute_value:
                return self._predict(X, v)
        
        print("Classification Error, Tree not complete.")
        return None

class TreeNode():
    def __init__(self, mass = None, attribute = None, attribute_value = 0, continuous_attribute = 0, number_leaf = 0, node_depth = 0):
        """
        Tree node class used in the Evidential Decision Tree

        Parameters
        -----
        mass : BBA
            Mass of the node
        attribute : int
            indice of the attribute
        attribute_value : float/string
            value of the attribute (string for categorical, float for numerical)
        continuous_attribute : int
            0: categorical, 1: numerical and <, 2: numerical and >=.
        number_leaf : int
            number of elements in the leaf

        Returns
        -----
        The instance of the class.
        """

        self.leafs = []

        self.mass = mass
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.continuous_attribute = continuous_attribute
        self.number_leaf = number_leaf
        self.node_depth = node_depth
    
    def max_depth(self, depth=1):
        maximum_depth = []
        for i in self.leafs:
            maximum_depth.append(i.max_depth(depth=depth + 1))
        
        if len(self.leafs) == 0:
            return depth

        return np.max(np.array(maximum_depth))

    def mean_samples_leafs(self):
        samples = []

        for i in self.leafs:
            childs = i.mean_samples_leafs()

            if isinstance(childs, int):
                samples.append(childs)
            else:
                for j in childs:
                    samples.append(j)
        
        if len(self.leafs) == 0:
            return self.number_leaf

        return samples

    def print_tree(self, depth=1):
        """
        Print the corresponding tree

        Parameters
        -----
        depth : Used for recursivity (do not use)
        """

        for i in self.leafs:
            if i.continuous_attribute == 0:
                print('|', '---' * depth, "Attribut", i.attribute, " : ", i.attribute_value)
            elif i.continuous_attribute == 1:
                print('|', '---' * depth, "Attribut", i.attribute, "<", i.attribute_value)
            elif i.continuous_attribute == 2:
                print('|', '---' * depth, "Attribut", i.attribute, ">=", i.attribute_value)
            i.print_tree(depth + 1)
        
        if len(self.leafs) == 0:
            print('    ' * depth, "N:", self.number_leaf,", Mass : ", np.around(self.mass, decimals=2))
    
    def add_leaf(self, node):
        """
        Add leaf to the node

        Parameters
        -----
        node : TreeNode
            Node
        """

        self.leafs.append(node)