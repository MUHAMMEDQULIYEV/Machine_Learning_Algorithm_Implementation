import numpy as np
from sklearn.metrics import accuracy_score

class KFoldCV:
    def __init__(self,model,X,y,k=5):
        self.model=model
        self.k=k
        self.X=X
        self.y=y
        self.scores=[]
    
    def splits_data(self):
        indicies=np.arange(len(self.X))
        np.random.shuffle(indicies)
        fold_size=len(self.X)//self.k
        fold=[indicies[i*fold_size:(i+1)*fold_size] for i in range(self.k)]
        return folds
    def validate(self):
        fold=self.splits_data()
        for i in range(self.k):
         validation_indicies=folds[i]
         training_indicies=np.concatenate([fold[j] for j in range(self.k) if j!=i])
         X_train, X_val = self.X[training_indices], self.X[validation_indices]
         y_train, y_val = self.y[training_indices], self.y[validation_indices]

         self.model.fit(X_train,y_train)
         predictions=self.model.predict(X_val)
         score=accuracy_score(y_val,predictions)
         self.scores.append(score)
         print(f"Fold {i+1}: Accuracy = {score:.4f}")


    
