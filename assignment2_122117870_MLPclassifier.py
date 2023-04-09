from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.pipeline import Pipeline

classifierMLP1 = Pipeline([
    ('scaler', MinMaxScaler()),
    ('pca', PCA(n_components=0.99, svd_solver='auto')),
    ('scaler2', MinMaxScaler()),
    ('selector', SelectKBest(k=25, score_func=chi2)),
    ('clf', MLPClassifier(max_iter=1000, early_stopping=True, tol=0.000001, activation='tanh',\
                         alpha=0.01, hidden_layer_sizes=(60), learning_rate='adaptive', solver='lbfgs',\
                         ))
])
