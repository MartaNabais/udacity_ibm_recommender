# IBM Article Recommender
## Business Understanding
In the IBM Watson Studio, there is a large collaborative community ecosystem of articles, 
datasets, articles, notebooks, and other AI and ML assets. 
Users of the system interact with all of this.

Recommender systems are particularly useful when an individual needs to choose an item from a potentially overwhelming 
number of items that a service may offer.

These systems typically leverage collaborative filtering, content-based filtering (also referred to as the 
personality-based approach), and sometimes knowledge-based systems. 

Collaborative filtering constructs a model based on
a user's past interactions, such as purchases or ratings, along with analogous decisions from fellow users, facilitating
predictions of items of potential interest to the user.

On the other hand, content-based filtering relies on discrete, pre-tagged item characteristics to suggest similar items,
effectively expanding the user's scope based on shared properties.

In this project, I have analyzed the interactions that users have with articles on the IBM Watson Studio platform, and make 
recommendations to them about new articles that they may like.

The project is contained within `Recommendations_with_IBM.ipynb
`

## Notebook Contents
1. Exploratory Data Analysis
2. Rank Based Recommendations
3. User-User Based Collaborative Filtering
4. Matrix Factorization
5. Conclusions

## Files Description
+ `Recommendations_with_IBM.ipynb`: Executed Jupyter Notebook containing the different recommender systems.
+ `data/articles_community.csv`: csv file containing article contents (e.g., full title, description).
+ `data/user-item-interactions.csv`: csv file containing user-article interactions.


