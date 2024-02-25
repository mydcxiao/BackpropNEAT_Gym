import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io  # Import io for handling byte streams
import sys
sys.path.append('../domain/')
sys.path.append('vis')
from domain.config import games
import gym
from domain import *   # Task environments
from neat_src import * # NEAT
from viewInd import *  # Graph visualization


def viewClassifier(ind, taskName, seed=None):
    """Visualize a 2D classification problem with decision boundary
       X: (n_samples, 2) - input features
       y: (n_samples, 1) - target labels
       pred_logits: (n_samples, 1) - predicted logits
    """
    task = GymTask(games[taskName])
    env = games[taskName]
    if isinstance(ind, str):
        ind = np.loadtxt(ind, delimiter=',') 
        wMat = ind[:,:-1]
        aVec = ind[:,-1]
    else:
        wMat = ind.wMat
        aVec = np.zeros((np.shape(wMat)[0]))
        
    # Create Graph
    nIn = env.input_size+1 # bias
    nOut= env.output_size
    G, layer= ind2graph(wMat, nIn, nOut)
    pos = getNodeCoord(G,layer,taskName)
    
    # TODO: merge graph visualization into one figure
    # Draw Graph
    # fig = plt.figure(figsize=(20,10), dpi=100)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), dpi=100)
    # ax1 = fig.add_subplot(122)
    drawEdge(G, pos, wMat, layer, ax1)
    nx.draw_networkx_nodes(G,pos,\
        node_color='lightblue',node_shape='o',\
        cmap='terrain',vmin=0,vmax=6, ax=ax1)
    drawNodeLabels(G,pos,aVec, ax1) 
    labelInOut(pos,env, ax1)
  
    ax1.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelleft=False,
        labelbottom=False) # labels along the bottom edge are off
    
    # ax2 = fig.add_subplot(111)
    # Prepare mesh and predictions as before
    task.env._generate_data(seed=seed)
    X, y = task.env.trainSet, task.env.target
    # Predict logits
    annOut = act(wMat, aVec, task.nInput, task.nOutput, X)
    action = selectAct(annOut, task.actSelect)
    pred = np.where(action > 0.5, 1, 0).reshape(-1, 1)
    test_acc = np.mean(pred == y)
    
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=1000),
                         np.linspace(X[:, 1].min(), X[:, 1].max(), num=1000))
    pred_contour = selectAct(act(wMat, aVec, task.nInput, task.nOutput, np.c_[xx.ravel(), yy.ravel()]), task.actSelect)
    pred_contour = np.where(pred_contour > 0.5, 1, 0).reshape(xx.shape)
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Plot setup as before
    ax2.contourf(xx, yy, pred_contour, alpha=0.8, levels=np.linspace(0, 1, 11), cmap=plt.cm.coolwarm)
    ax2.scatter(X[pos_idx, 0], X[pos_idx, 1], c='r', marker='o', edgecolors='k')
    ax2.scatter(X[neg_idx, 0], X[neg_idx, 1], c='b', marker='o', edgecolors='k')
    # train_accuracy = classifier.score(X_train, y_train)
    # plt.text(xx.min() + 0.3, yy.min() + 0.3, f'Train accuracy = {train_accuracy * 100:.1f}%', fontsize=12)
    ax2.text(xx.min() + 0.3, yy.min() + 0.7, f'Test accuracy = {test_acc * 100:.1f}%', fontsize=12)
    ax2.set_title('2D Classification with Decision Boundary')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_xlim(xx.min(), xx.max())
    ax2.set_ylim(yy.min(), yy.max())

    # Save the plot to a bytes buffer
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')  # Save the figure to the buffer
    # buf.seek(0)  # Seek to the start of the stream

    # Use PIL to save the image from the buffer
    # img = Image.open(buf)
    # img.save('/mnt/data/classification_plot.png')  # Change this to your desired path and file name

    # Close the buffer
    # buf.close()

    # Show the plot as well if needed
    # plt.show()
    
    return fig, ax1, ax2