import numpy as np
from numpy.linalg import norm
import pandas as pd

from random import normalvariate
from math import sqrt


# Read data from the CSV file into a Pandas DataFrame
df = pd.read_csv("./ml-latest-small/ratings.csv")
movie_ratings = df.groupby(['userId', 'movieId'])['rating'].first().unstack(fill_value=0.0)

movies = pd.read_csv("./ml-latest-small/movies.csv")

def random_unit_vector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    the_norm = sqrt(sum(x * x for x in unnormalized))
    return [x / the_norm for x in unnormalized]


def svd_1d(A, epsilon=1e-10):
    ''' The one-dimensional SVD '''

    n, m = A.shape
    x = random_unit_vector(min(n,m))
    last_v = None
    current_v = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        last_v = current_v
        current_v = np.dot(B, last_v)
        current_v = current_v / norm(current_v)

        if abs(np.dot(current_v, last_v)) > 1 - epsilon:
            return current_v


def svd(A, k=None, epsilon=1e-10):
    '''
        Compute the singular value decomposition of a matrix A
        using the power method. A is the input matrix, and k
        is the number of singular values you wish to compute.
        If k is None, this computes the full-rank decomposition.
    '''
    A = np.array(A, dtype=float)
    n, m = A.shape
    svd_so_far = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrix_for1D = A.copy()

        for singular_value, u, v in svd_so_far[:i]:
            matrix_for1D -= singular_value * np.outer(u, v)

        if n > m:
            v = svd_1d(matrix_for1D, epsilon=epsilon)  # next singular vector
            u_unnormalized = np.dot(A, v)
            sigma = norm(u_unnormalized)  # next singular value
            u = u_unnormalized / sigma
        else:
            u = svd_1d(matrix_for1D, epsilon=epsilon)  # next singular vector
            v_unnormalized = np.dot(A.T, u)
            sigma = norm(v_unnormalized)  # next singular value
            v = v_unnormalized / sigma

        svd_so_far.append((sigma, u, v))

    singular_values, us, vs = [np.array(x) for x in zip(*svd_so_far)]
    return us.T, singular_values, vs


print('Waiting')
U, S, Vt = svd(movie_ratings, k=50)

sigma_diag_matrix=np.diag(S)
all_user_predicted_ratings = np.dot(np.dot(U, sigma_diag_matrix), Vt)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = movie_ratings.columns, index=movie_ratings.index)


def get_high_recommended_movies(user_id, max_recommend=100):
    movies_rated_by_user = movie_ratings.loc[user_id]
    movies_high_rated_by_user =  movies_rated_by_user[movies_rated_by_user > 2].index.tolist()
    
    movies_recommended_for_user = preds_df.loc[user_id]
    movies_high_recommend_for_user = movies_recommended_for_user[movies_recommended_for_user > 2].sort_values(ascending=False).index.tolist()
    
    recommend_id = []
    for movie_id in movies_high_recommend_for_user:
        if movie_id not in movies_high_rated_by_user:
            recommend_id.append(movie_id)
        if len(recommend_id) >= max_recommend:
            break
            
    recommend_name = []
    for movie_id in recommend_id:
        movie_name = ' '.join(str(movies.loc[movies['movieId'] == movie_id]['title']).split('\n')[0].split()[1:])
        recommend_name.append(movie_name)
        
    return recommend_name


# user_id = 156
while True:
    try:
        user_id = int(input('Enter user_id: '))
        recommend_movies = get_high_recommended_movies(user_id)

        print(f'user_id = {user_id}\n')
        print('recommendation Movies: ')
        for movie in recommend_movies:
            print(movie)
        print()
    
    except:
        break

