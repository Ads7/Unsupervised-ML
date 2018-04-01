
# <font face="Luxi Sans">HW2 KMEANS and Gaussian Mixtures  
</font>

* * *

DATATSET <span style="font-weight: bold;">: [SpamBase](https://archive.ics.uci.edu/ml/datasets/spambase): emails (54-feature vectors) classified as spam/nospam</span>

DATATSET <span style="font-weight: bold;">: 20 NewsGroups : news articles</span>

DATATSET <span style="font-weight: bold;">: MNIST : 28x28 digit B/W images</span>

DATATSET <span style="font-weight: bold;">: FASHION : 28x28 B/W images</span>

https://en.wikipedia.org/wiki/MNIST_database  
http://yann.lecun.com/exdb/mnist/  
https://www.kaggle.com/zalando-research/fashionmnist  

### PROBLEM 1: KMeans Theory

Given Kmeans Objective discussed in class with Euclidian distance  
![some text](kmeans_obj.jpeg)  
A) prove that E step update on membership (\pi) achieves the minimum objective given the current centroids( \mu)  
B) prove that M step update on centroids (\mu) achievess the minimum objective given the current memberships( \pi)  
C) Explain why KMeans has to stop (converge), but not necessarily to the global minimum objective value.

### PROBLEM 2 : KMeans on data

Using Euclidian distance or dot product similarity (choose one per dataset, you can try other similarity metrics),  
A) run KMeans on the MNIST Dataset, try K=10  
B) run KMeans on the FASHION Dataset, try K=10  
C) run KMeans on the 20NG Dataset, try K=20  

For all three datasets, evaluate the KMeans objective for a higher K (for example double) or smaller K(for example half).  
For all three datasets, evaluate external clustering performance using data labels and performance metrics Purity and Gini Index (see [A] book section 6.9.2).  

### PROBLEM 3 : Gaussian Mixture on toy data

You are required to implemet the main EM loop, but can use math API/functions provided by your language to calculate normal densities, covariance matrix, etc.  
A) The gaussian 2-dim data on file  [2gaussian.txt](2gaussian.txt)  has been generated  using a mixture  of  two Gaussians, each  2-dim, with the parameters below. Run the EM algorithm with random initial values to recover the parameters.  
<span style="font-family: monospace;">mean_1 [3,3]); cov_1 = [[1,0],[0,3]]; n1=2000 points</span>  
<span style="font-family: monospace;">mean_2 =[7,4]; cov_2 = [[1,0.5],[0.5,1]]; ; n2=4000 points  
</span>You should obtain a result visually [like this](23.png) (you dont necessarily have to plot it)<span style="font-family: monospace;">  

</span>B) Same problem for 2-dim data on file [3gaussian.txt](3gaussian.txt) , generated using a mixture of three Gaussians. Verify your  findings against the true parameters used generate the data below.  
<span style="font-family: monospace;">mean_1 = [3,3] ; cov_1 = [[1,0],[0,3]]; n1=2000</span>  
<span style="font-family: monospace;">mean_2 = [7,4] ; cov_2 = [[1,0.5],[0.5,1]] ; n2=3000</span>  
<span style="font-family: monospace;">mean_3 = [5,7] ; cov_3 = [[1,0.2],[0.2,1]]    ); n3=5000</span>  
<span style="font-family: monospace;"></span>  
Additional notes helpful for implementing Gaussian Mixtures:  
[https://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/](https://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/)  
[http://andrewgelman.com/2016/06/11/log-sum-of-exponentials/](http://andrewgelman.com/2016/06/11/log-sum-of-exponentials/)  
[https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/](https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/)  

### PROBLEM 4 : Gaussian Mixture on real data

Run EM to obtain a Gaussian Mixture on FASHION dataset (probably won't work) and on SPAMBASE dataset (should work). Use a library/package (such as scikit-learn) and at first use the option that imposes a diagonal covariance matrix.  
Sampling data might be necessary to complete the run.


# <font face="Luxi Sans">HW2B Clustering: DBSCAN, Hierarchical Clustering  </font>

Make sure you check the [<font face="Luxi Sans">syllabus</font>](../../html/schedulen.html) for the due date. Please use the notations adopted in class, even if the problem is stated in the book using a different notation.

We are not looking for very long answers (if you find yourself writing more than one or two pages of typed text per problem, you are probably on the wrong track). Try to be concise; also keep in mind that good ideas and explanations matter more than exact details.

Submit all code files Dropbox (create folder HW1 or similar name). Results can be pdf or txt files, including plots/tabels if any.  

"Paper" exercises: submit using Dropbox as pdf, either typed or scanned handwritten.  

* * *

DATATSET <span style="font-weight: bold;">: 20 NewsGroups : news articles</span>

DATATSET <span style="font-weight: bold;">: MNIST : digit images</span>

https://en.wikipedia.org/wiki/MNIST_database  
http://yann.lecun.com/exdb/mnist/  

DATATSET <span style="font-weight: bold;">: [FASHION](https://www.kaggle.com/zalando-research/fashionmnist) : 28x28 B/W images</span>

DATATSET <span style="font-weight: bold;">: [UCI/Household](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption#)</span>

### PROBLEM 5: DBSCAN on toy-neighborhood data

You are to cluster, and visualize, a small dataset using DBSCAN epsilon = 7.5, MinPts = 3). You have been provided a file, [dbscan.csv](dbscan.csv), that has the following columns for each point in the dataset:

*   cluster originally empty, provided for your convenience pt a unique id for each data point
*   x point x-coordinate
*   y point y-coordinate
*   num neighbors number of neighbors, according to the coordinates above neighbors the id’s of all neighbors within

As you can see, a tedious O(n^2) portion of the work has been done for you. Your job is to execute, point-by-point, the DBSCAN algorithm, logging your work.  

### PROBLEM 6: DBSCAN on toy raw data

Three toy 2D datasets are provided (or they can be obtained easily with scikit learn) [circles](circle.csv); [blobs](blobs.csv), and [moons](moons.csv). Run your own implementaion of DBSCAN on these, in two phases.  

### PROBLEM 7: DBSCAN on real data

Run the DBSCAN algorithm on the 20NG dataset, and on the FASHION dataset, and the HouseHold dataset (see papers), and evaluate results. You need to implement both phases (1) neighborhoods creation, (2) DBSCAN.  
Explain why/when it works, and speculate why/when not. You need to trial and error for parameters epsilon and MinPts  

[DBSCAN Revisited, Revisited: Why and How You Should (Still) Use DBSCAN](../notes_slides/revisitofrevisitDBSCAN.pdf)  
[DBSCAN Revisited:Mis-Claim, Un-Fixability, and Approximation](../notes_slides/sigmod15-dbscan.pdf)  

EXTRA CREDIT: Using class labels (cheating), try to remove/add points in curate the set for better DBSCAN runs  

### PROBLEM 8: Hierarchical Clustering

Use a library to execute hierarchical clustering on MNIST dataset, evaluate the clusters.
