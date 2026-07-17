"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree 
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        # calculate the height
        height = np.amax(self.points[:, 2]) # indexing, flatteend to 1D array
        self.feature.append(height)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)  # slicing, 2D array
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)
        # Helps distinguish compact objects (poles, trees) from elongated ones (walls, fences)

        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        dist, idx = kd_tree_3d.query(top, k=k_top, return_distance=True)
        dist = dist.flatten()
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, vector = np.linalg.eig(cov) # vector is eigenvectors
        # sort eigenvalues and eigenvectors together
        sort_indices = w.argsort()
        w = w[sort_indices]
        vector = vector[:, sort_indices]

        # calculate the linearity and sphericity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        planarity = (w[1] - w[0]) / (w[2] + 1e-5) # an addition of planarity calculation. DOI:10.1016/j.isprsjprs.2015.01.016
        self.feature += [linearity, sphericity, planarity]

        w_sum = np.sum(w) + 1e-5
        e = w / w_sum

        # eigenentropy: estimation of the order/disorder of 3D points within local 3D neighborhood
        eigenentropy = - 1.0 * np.sum(e * np.log(e + 1e-5))

        # calculate anisotropy, omnivariance, and curvature
        anisotropy = (w[2] - w[0]) / (w[2] + 1e-5)
        omnivariance = (abs(w[0] * w[1] * w[2])) ** (1.0 / 3.0)
        curvature = w[2] / (w[0] + w[1] + w[2] + 1e-5)
        self.feature += [eigenentropy, anisotropy, omnivariance, curvature]

        # verticality : high -> curved surfaces. Curbs, domes, rounded obstacles
        nz = vector[:,0]
        verticality = 1 - abs(nz[2])
        height_diff = np.amax(self.points[:, 2]) - 1.0 * np.amin(self.points[:, 2])

        geom = np.array([linearity, planarity, sphericity])
        geom_norm = geom / (np.sum(geom) + 1e-5)
        e_dim = - 1.0 * np.sum(geom_norm * np.log(geom_norm + 1e-5))

        furthest_dist = dist[-1]
        lpd = (k_top + 1.0) / ((4/3)*math.pi*(furthest_dist**3) + 1e-5)  # local point density

        self.feature += [verticality, height_diff, e_dim, lpd] 


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = 'ID,label,height,root_density,area,shape_index,linearity,sphericity,planarity,' \
                  'eigenentropy,anisotropy,omnivariance,curvature,verticality,height_diff,e_dim,' \
                  'local_point_density'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree'] 

    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 3], X[100*i:100*(i+1), 4], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel('x1: root density')
    ax.set_ylabel('x2: area')
    ax.legend()
    plt.show()


def SVM_classification(X_train , X_test , y_train , y_test):
    """
    Conduct SVM classification
        X: features
        y: labels
    use the implementation of the classifiers from Scikit-Learn. For the SVM classifier, please try different Kernel 
    functions and then recommend the most promising one (and justify your choice in the report). 
    For both classifiers (i.e., SVM, RF), try different combinations of hyper-parameters and find the best model with the 
    highest performance.

    """
    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # define the hyperparam grid:
    #   - kernel
    #   - penalty parameter
    #   - kernel coefficient (note that this is ignored for the linear one)
    param_grid = {
        'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1]
    }

    print("\nTuning SVM hyperparameters. This might take a while :) ...")

    # cv=5 means that we do a 5-fold cross-validation
    # refit=True means to automatically keep the best model
    svm_grid = GridSearchCV(svm.SVC(), param_grid, cv=5, refit=True, verbose=0)
    svm_grid.fit(X_train_scaled, y_train)

    # print the best parameters
    print("Best SVM Parameters found: {0}".format(svm_grid.best_params_))

    # predict only using the best model
    pred = svm_grid.predict(X_test_scaled)

    # calculate and print metrics
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    conf = confusion_matrix(y_test, pred)

    print("\n--- Best SVM Performance ---\n"
          "Accuracy : {0:.2f}\n"
          "F1 Score : {1:.2f}\n"
          "Confusion Matrix:\n"
          "{2}\n".format(acc, f1, conf))

    # return the trained and tuned model
    return svm_grid.best_estimator_


def RF_classification(X_train , X_test , y_train , y_test):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    # define the hyperparam grid:
    #   - number of trees in the forest
    #   - maximum depth of the trees
    #   - minimum samples required to split a node
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    print("\nTuning Random Forest hyperparameters...")

    # cv=5 means that we do a 5-fold cross-validation
    # refit=True means to automatically keep the best model
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=101), param_grid, cv=5, refit=True, verbose=0)
    rf_grid.fit(X_train, y_train)

    # print the best parameters
    print("Best RF Parameters found: {0}".format(rf_grid.best_params_))

    # predict only using the best model
    pred = rf_grid.predict(X_test)

    # calculate and print metrics
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    conf = confusion_matrix(y_test, pred)

    print("\n--- Best RF Performance ---\n"
          "Accuracy : {0:.2f}\n"
          "F1 Score : {1:.2f}\n"
          "Confusion Matrix:\n"
          "{2}\n".format(acc, f1, conf))

    # return the trained and tuned model
    return rf_grid.best_estimator_


def plot_custom_learning_curve(X, y, best_svm, best_rf):
    """
    Generate learning curves for SVM and RF.
    Varies train_size from 0.1 to 0.9 (1:9 to 9:1 splits).
    """
    print("\nGenerating Learning Curves...")

    # split ratios
    train_sizes = np.arange(0.1, 1.0, 0.1)

    # store results as lists
    svm_train_scores, svm_test_scores = [], []
    rf_train_scores, rf_test_scores = [], []
    num_train_samples = []

    for size in train_sizes:
        # split the data at the current ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=size, random_state=42)

        # record the number of samples in this set
        num_train_samples.append(len(X_train))

        # scale features (for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # train the models on the current subset
        best_svm.fit(X_train_scaled, y_train)
        best_rf.fit(X_train, y_train)

        # evaluate SVM
        svm_train_scores.append(accuracy_score(y_train, best_svm.predict(X_train_scaled)))
        svm_test_scores.append(accuracy_score(y_test, best_svm.predict(X_test_scaled)))

        # evaluate RF
        rf_train_scores.append(accuracy_score(y_train, best_rf.predict(X_train)))
        rf_test_scores.append(accuracy_score(y_test, best_rf.predict(X_test)))

    # plotting
    plt.figure(figsize=(12, 5))

    # subplot 1: SVM
    plt.subplot(1, 2, 1)
    plt.plot(num_train_samples, svm_train_scores, marker='o', label='Train Accuracy', color='blue')
    plt.plot(num_train_samples, svm_test_scores, marker='s', label='Test Accuracy', color='orange')
    plt.title('SVM Learning Curve')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # subplot 2: RF
    plt.subplot(1, 2, 2)
    plt.plot(num_train_samples, rf_train_scores, marker='o', label='Train Accuracy', color='blue')
    plt.plot(num_train_samples, rf_test_scores, marker='s', label='Test Accuracy', color='orange')
    plt.title('Random Forest Learning Curve')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

class feature_selection:
    @staticmethod
    def within_class_sm(X, y):
        n_total = X.shape[0]
        n_features = X.shape[1]
        S_w = np.zeros((n_features, n_features))
        unique_labels = np.unique(y)
        for label in unique_labels:
            # extract objects belonging to class
            class_samples = X[y == label]
            nk = class_samples.shape[0]
            cov_mat = np.cov(class_samples, rowvar=False)
            S_w += (nk/n_total)*cov_mat
        return S_w
    
    @staticmethod
    def between_class_sm(X, y):
        n_total = X.shape[0]
        n_features = X.shape[1]
        unique_labels = np.unique(y)
        S_b = np.zeros((n_features, n_features))
        mean_total = np.mean(X, axis=0)
        for label in unique_labels:
            # extract objects belonging to class
            class_samples = X[y == label]
            nk = class_samples.shape[0]
            mean_class = np.mean(class_samples, axis=0)
            mean_diff = (mean_class - mean_total)
            S_b  += (nk/n_total)*np.outer(mean_diff,mean_diff)
        return S_b

if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'pointclouds-500/pointclouds-500'
   
    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, test_size=0.40, random_state=42)

    print('\nStart Feature Selection')
    S_w = feature_selection.within_class_sm(X_train, y_train)
    S_b = feature_selection.between_class_sm(X_train, y_train)
    ratio = np.diag(S_b) / (np.diag(S_w) + 1e-5)

    feature_names = np.array([
        'height', 'root_density', 'area', 'shape_index', 'linearity', 'sphericity', 'planarity',
        'eigenentropy', 'anisotropy', 'omnivariance', 'curvature', 'verticality', 'height_diff',
        'e_dim', 'local_point_density'
    ])
    for i, r in enumerate(ratio):
        print(f"Feature {i} ({feature_names[i]}): J = {r:.4f}")

    # find the indices of the top 4 features (so the last 4 indexes, as they are sorted ascendingly)
    top_4_indices = np.argsort(ratio)[-4:]
    print(f"\nSelecting Top 4 Features: {feature_names[top_4_indices]}")

    X_train_selected = X_train[:, top_4_indices]
    X_test_selected = X_test[:, top_4_indices]

    # SVM classification
    print('Start SVM classification')
    tuned_SVM = SVM_classification(X_train_selected, X_test_selected, y_train, y_test)

    # RF classification
    print('Start RF classification')
    tuned_RF = RF_classification(X_train_selected, X_test_selected, y_train, y_test)

    # learning curve
    X_selected_full = X[:, top_4_indices]
    plot_custom_learning_curve(X_selected_full, y, tuned_SVM, tuned_RF)

"""
references
1. Semantic point cloud interpretation based on optimal neighborhoods, relevant features and efficient classifiers
for planarity, etc except for verticality from paper 2
2. STREAMED VERTICAL RECTANGLE DETECTION IN TERRESTRIAL LASER SCANS FOR FACADE DATABASE PRODUCTION 
(https://isprs-annals.copernicus.org/articles/I-3/99/2012/isprsannals-I-3-99-2012.pdf)
-> verticality
"""
