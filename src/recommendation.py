import streamlit as st
import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader  # package that helps transform your data to machine learning readiness
from sklearn.cluster import KMeans
import os

# Get directory paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(root_dir, "data/{}")

# Get data
movies_df = pd.read_csv('data/recommendation/movies.csv')
ratings_df = pd.read_csv('data/recommendation/ratings.csv')


# Note: This isn't 'good' practice, in a MLops sense, but we'll roll with this since the data
# is already loaded in memory.
class Loader(Dataset):
    def __init__(self):
        self.ratings = ratings_df.copy()

        # Extract all user IDs and movie IDs
        users = ratings_df.userId.unique()
        movies = ratings_df.movieId.unique()

        # --- Producing new continuous IDs for users and movies ---

        # Unique values : index
        self.userid2idx = {o: i for i, o in enumerate(users)}
        self.movieid2idx = {o: i for i, o in enumerate(movies)}

        # Obtained continuous ID for users and movies
        self.idx2userid = {i: o for o, i in self.userid2idx.items()}
        self.idx2movieid = {i: o for o, i in self.movieid2idx.items()}

        # return the id from the indexed values as noted in the lambda function down below.
        self.ratings.movieId = ratings_df.movieId.apply(lambda x: self.movieid2idx[x])
        self.ratings.userId = ratings_df.userId.apply(lambda x: self.userid2idx[x])

        self.x = self.ratings.drop(['rating', 'timestamp'], axis=1).values
        self.y = self.ratings['rating'].values
        self.x, self.y = torch.tensor(self.x), torch.tensor(
            self.y)  # Transforms the data to tensors (ready for torch models.)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.ratings)


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors)  # think of this as a lookup table for the input.
        # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors)  # think of this as a lookup table for the input.
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.item_factors.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        # matrix multiplication
        users, items = data[:, 0], data[:, 1]
        return (self.user_factors(users) * self.item_factors(items)).sum(1)

    # def forward(self, user, item):
    # 	# matrix multiplication
    #     return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)


def app():
    st.write('The dimensions of movies dataframe are:', movies_df.shape,
             '\nThe dimensions of ratings dataframe are:', ratings_df.shape)

    st.title("Work in progress")

    st.write(movies_df.head())
    st.write(ratings_df.head())

    # Movie ID to movie name mapping
    movie_names = movies_df.set_index('movieId')['title'].to_dict()
    n_users = len(ratings_df.userId.unique())
    n_items = len(ratings_df.movieId.unique())
    st.write("Number of unique users:", n_users)
    st.write("Number of unique movies:", n_items)
    st.write("The full rating matrix will have:", n_users * n_items, 'elements.')
    st.write('----------')
    st.write("Number of ratings:", len(ratings_df))
    st.write("Therefore: ", len(ratings_df) / (n_users * n_items) * 100, '% of the matrix is filled.')
    st.write("We have an incredibly sparse matrix to work with here.")
    st.write("And... as you can imagine, as the number of users and products grow, "
             "the number of elements will increase by n*2")
    st.write("You are going to need a lot of memory to work with global scale... "
             "storing a full matrix in memory would be a challenge.")
    st.write("One advantage here is that matrix factorization can realize the rating matrix implicitly, "
             "thus we don't need all the data")
    st.write("F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. "
             "ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. "
             "<https://doi.org/10.1145/2827872>")

    num_epochs = 128
    cuda = torch.cuda.is_available()

    st.write("Is running on GPU:", cuda)

    model = MatrixFactorization(n_users, n_items, n_factors=8)
    st.write(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            st.write(name, param.data)
    # GPU enable if you have a GPU...
    if cuda:
        model = model.cuda()

    # MSE loss
    loss_fn = torch.nn.MSELoss()

    # ADAM optimizier
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train data
    train_set = Loader()
    train_loader = DataLoader(train_set, 128, shuffle=True)

    for it in tqdm(range(num_epochs)):
        losses = []
        for x, y in train_loader:
            if cuda:
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()
                outputs = model(x)
                loss = loss_fn(outputs.squeeze(), y.type(torch.float32))
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
        st.write("iter #{}".format(it), "Loss:", sum(losses) / len(losses))

    trained_movie_embeddings = model.item_factors.weight.data.cpu().numpy()
    len(trained_movie_embeddings)  # unique movie factor weights

    # Fit the clusters based on the movie weights
    kmeans = KMeans(n_clusters=10, random_state=0).fit(trained_movie_embeddings)

    '''It can be seen here that the movies that are in the same cluster tend to have
    similar genres. Also note that the algorithm is unfamiliar with the movie name
    and only obtained the relationships by looking at the numbers representing how
    users have responded to the movie selections.'''
    for cluster in range(10):
        st.write("Cluster #{}".format(cluster))
        movs = []
        for movidx in np.where(kmeans.labels_ == cluster)[0]:
            movid = train_set.idx2movieid[movidx]
            rat_count = ratings_df.loc[ratings_df['movieId'] == movid].count()[0]
            movs.append((movie_names[movid], rat_count))
        for mov in sorted(movs, key=lambda tup: tup[1], reverse=True)[:10]:
            st.write("\t", mov[0])


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
