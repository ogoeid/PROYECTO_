import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
X = iris.data[:, :2]          # sólo las dos primeras características
y = iris.target

# Nos quedamos con dos clases: setosa (0) vs versicolor (1)
mask = y < 2
X, y = X[mask], y[mask]
y = np.where(y == 0, -1, 1)   # Perceptrón usa etiquetas -1 y 1
plt.scatter(X[y==-1,0], X[y==-1,1], color='blue', label='setosa')
plt.scatter(X[y==1,0], X[y==1,1], color='red', label='versicolor')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris reducido (2 clases, 2 feats)')
plt.legend()
plt.show()
# Normalización
scaler = StandardScaler()
X = scaler.fit_transform(X)

# División 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
class Perceptron:
    def __init__(self, lr=0.01, n_epochs=50):
        self.lr = lr
        self.n_epochs = n_epochs
        self.errors_ = []          # errores por época
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w_ = np.zeros(1 + n_features)  # incluye sesgo
        for _ in range(self.n_epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
ppn = Perceptron(lr=0.01, n_epochs=50)
ppn.fit(X_train, y_train)
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marcador y mapa de colores
    markers = ('s', 'x')
    colors = ('blue', 'red')
    cmap = plt.cm.ListedColormap(colors[:len(np.unique(y))])
    
    # superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='k')

plot_decision_regions(X_test, y_test, classifier=ppn)
plt.xlabel('Sepal length (std)')
plt.ylabel('Sepal width (std)')
plt.legend(loc='upper left')
plt.title('Frontera de decisión – conjunto de prueba')
plt.show()
def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marcador y mapa de colores
    markers = ('s', 'x')
    colors = ('blue', 'red')
    cmap = plt.cm.ListedColormap(colors[:len(np.unique(y))])
    
    # superficie de decisión
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(X[y == cl, 0], X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='k')

plot_decision_regions(X_test, y_test, classifier=ppn)
plt.xlabel('Sepal length (std)')
plt.ylabel('Sepal width (std)')
plt.legend(loc='upper left')
plt.title('Frontera de decisión – conjunto de prueba')
plt.show()
y_pred = ppn.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Precisión en test: {accuracy:.2%}')
