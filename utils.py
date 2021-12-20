from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pandas as pd
def plot_confusion_matrix(TestGen, model):
    """[function to plot the confusion matrix]

    Args:
        TestGen ([type]): [data loader]
        model ([object]): [the trained model object (the whole class)]
    """
    true = list(map(lambda x : TestGen.inv_ClsIdxDic[x],TestGen.get_true() ))
    pred = list(map(lambda x : TestGen.inv_ClsIdxDic[x], np.argmax(model.model.predict(TestGen), axis = 1)))

    confusion = confusion_matrix(true, pred)


    
    df = pd.DataFrame(confusion, index = list(TestGen.ClsIdxDic.keys()),
                    columns = list(TestGen.ClsIdxDic.keys())) 

    fig, ax = plt.subplots(figsize=(10,10))
    plt.xlabel("Predicted")
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(df.shape[0]) + 0.5, minor=False)
    plt.Axes.set_title(ax,"prediction", fontsize=15)
    plt.ylabel("Actual", fontsize=15)
    plt.ylim(0,TestGen.n_classes)
    plt.xlim(TestGen.n_classes,0)
    sn.heatmap(df, annot=True,fmt="d", cmap='Greens', linecolor='black', linewidths=1)
    plt.ylabel('Actual', rotation=0, va='center')
    plt.yticks(rotation=0)
    plt.show()

def export_csv_predictions(TestGen, model):
    """[function to plot the confusion matrix]

    Args:
        TestGen ([type]): [data loader]
        model ([object]): [the trained model object (the whole class)]
    """
    true = list(map(lambda x : TestGen.inv_ClsIdxDic[x],TestGen.get_true() ))
    pred = list(map(lambda x : TestGen.inv_ClsIdxDic[x], np.argmax(model.model.predict(TestGen), axis = 1)))
    title = TestGen.X

    export_dic={'Title': title,
                'True Category':true,
                'Predicted Category':pred,
    }
    DF = pd.DataFrame(export_dic).to_csv('Predictions.csv')