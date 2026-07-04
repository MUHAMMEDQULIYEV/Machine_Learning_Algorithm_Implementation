import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
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
    def cross_validate(self):
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
         
         



        average_score=np.mean(self.scores)
        print(f"\nAverage Accuracy across {self.k} folds: {average_score:.4f}")
        return average_score 



    def plot_folds(X,y,folds):
        colors = ['red', 'blue', 'green', 'orange', 'purple']


        for i,fold in enumerate(folds):
            X_fold=X[fold]
            plt.scatter(X_fold[:,0],X_fold[:,1],color=colors[i],label=f"Fold {i+1}")

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Visualization of K-Folds")
        plt.legend()
        plt.show()
    

        
    
        


class KFoldCVWithConfusionMatrix(KFoldCV):
    def __init__(self, model, X, y, k=5):
        super().__init__(model, X, y, k)

    def cross_validate(self):
        folds = self.split_data()
        all_conf_matrices = []

        for i in range(self.k):
            validation_indices = folds[i]
            training_indices = np.concatenate([folds[j] for j in range(self.k) if j != i])

            X_train, X_val = self.X[training_indices], self.X[validation_indices]
            y_train, y_val = self.y[training_indices], self.y[validation_indices]

            # Train and validate the model
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_val)
            score = accuracy_score(y_val, predictions)
            self.scores.append(score)

            # Generate confusion matrix for each fold
            conf_matrix = confusion_matrix(y_val, predictions)
            all_conf_matrices.append(conf_matrix)
            print(f"Fold {i+1}: Accuracy = {score:.4f}")
            self.plot_confusion_matrix(conf_matrix, i)

        # Calculate and return average score
        average_score = np.mean(self.scores)
        print(f"\nAverage Accuracy across {self.k} folds: {average_score:.4f}")
        return average_score, all_conf_matrices

    def plot_confusion_matrix(self, conf_matrix, fold_idx):
        """
        Plots confusion matrix for a given fold.
        """
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
        plt.title(f"Confusion Matrix - Fold {fold_idx+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

    
