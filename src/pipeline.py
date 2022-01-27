from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


pipe0 = Pipeline([
    ('scale',StandardScaler()),
])
